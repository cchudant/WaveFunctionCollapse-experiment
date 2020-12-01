use std::fmt::Debug;

use image::imageops::{flip_horizontal, rotate90};
use image::{DynamicImage, GenericImageView, ImageBuffer, Pixel};

#[derive(Debug, Copy, Clone)]
pub enum Symmetries {
    None,
    Mirror,
    Rotate,
    RotateMirror,
}

impl Symmetries {
    pub fn mirror(self) -> bool {
        match self {
            Symmetries::None => false,
            Symmetries::Mirror => true,
            Symmetries::Rotate => false,
            Symmetries::RotateMirror => true,
        }
    }
    pub fn rotate(self) -> bool {
        match self {
            Symmetries::None => false,
            Symmetries::Mirror => false,
            Symmetries::Rotate => true,
            Symmetries::RotateMirror => true,
        }
    }
}

use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;

#[derive(Debug, Clone)]
pub struct OverlappingModel {
    wave: Box<[bool]>,
    sums_of_ones: Box<[isize]>,
    entropies: Box<[f64]>,
    rng: ChaCha20Rng,
    observed: Option<Box<[usize]>>,
    weights: Box<[f64]>,
    compatible: Box<[isize]>,
    stack: Vec<((usize, usize), usize)>,
    sums_of_weights: Box<[f64]>,
    sums_of_weights_log_weights: Box<[f64]>,
    weight_log_weights: Box<[f64]>,
    propagator: Box<[Box<[usize]>]>,
    big_n: usize,
    t_count: usize,
    width: usize,
    height: usize,
    periodic: bool,
    distribution: Box<[f64]>,
    sum_of_weights: f64,
    sum_of_weight_log_weights: f64,
    starting_entropy: f64,
    patterns: Box<[ImageBuffer<image::Rgba<u8>, Vec<u8>>]>,
}

const DX: [isize; 4] = [-1, 0, 1, 0];
const DY: [isize; 4] = [0, 1, 0, -1];
const OPPOSITE: [usize; 4] = [2, 3, 0, 1];

impl OverlappingModel {
    pub fn new(
        img: &DynamicImage,
        big_n: usize,
        height: usize,
        width: usize,
        periodic_input: bool,
        periodic_output: bool,
        symmetries: Symmetries,
        // ground: i32,
        seed: u64,
    ) -> OverlappingModel {
        let in_width = img.width() as usize;
        let in_height = img.height() as usize;

        // let mut colors = HashSet::new();

        // for y in 0..height {
        //     for x in 0..width {
        //         let rgb = img.get_pixel(x as u32, y as u32);
        //         colors.insert(rgb);
        //     }
        // }

        // let color_count = colors.len();
        // let big_w = color_count.pow((big_n * big_n) as u32);

        let max_y = if periodic_input {
            in_height
        } else {
            in_height - big_n + 1
        };
        let max_x = if periodic_input {
            in_width
        } else {
            in_width - big_n + 1
        };

        let mut ps = Vec::new();
        for y in 0..max_y {
            for x in 0..max_x {
                let add = ps.len();
                ps.push(sample_image(img, x, y, big_n, big_n, in_width, in_height));
                if symmetries.mirror() {
                    ps.push(flip_horizontal(&ps[add]));
                }
                if symmetries.rotate() {
                    ps.push(rotate90(&ps[add]));
                    if symmetries.mirror() {
                        ps.push(flip_horizontal(&ps[add + 2]));
                    }
                    ps.push(rotate90(&ps[add + 2]));
                    if symmetries.mirror() {
                        ps.push(flip_horizontal(&ps[add + 4]));
                    }
                    ps.push(rotate90(&ps[add + 4]));
                    if symmetries.mirror() {
                        ps.push(flip_horizontal(&ps[add + 6]));
                    }
                }
            }
        }

        // ps.iter()
        //     .enumerate()
        //     .for_each(|(u, p)| p.save(format!("ps {}.png", u)).unwrap());

        ps.reverse();

        let mut weights = Vec::new();
        let mut patterns = Vec::new();
        while let Some(p) = ps.pop() {
            let mut weight: f64 = 1.0;

            let mut i = 0;
            while i < ps.len() {
                if &ps[i] == &p {
                    ps.remove(i);
                    weight += 1.0;
                } else {
                    i += 1
                }
            }

            patterns.push(p);
            weights.push(weight);
        }

        // println!("{:?}", weights);
        // patterns
        //     .iter()
        //     .zip(&weights)
        //     .enumerate()
        //     .for_each(|(i, (t, w))| t.save(format!("test{} --- {}.png", i, w)).unwrap());

        let t_count = weights.len();

        let mut propagator = vec![Vec::new(); t_count * 4];
        for d in 0..4 {
            for t in 0..t_count {
                for t2 in 0..t_count {
                    if is_neighbor(&patterns[t], &patterns[t2], DX[d], DY[d]) {
                        propagator[t * 4 + d].push(t2);
                    }
                }
            }
        }

        let propagator = propagator
            .into_iter()
            .map(|vec| vec.into_boxed_slice())
            .collect::<Vec<_>>();

        let mut weight_log_weights = vec![0.0; t_count];
        let mut sum_of_weights = 0.0;
        let mut sum_of_weight_log_weights = 0.0;
        for (t, &weight) in weights.iter().enumerate() {
            weight_log_weights[t] += weight * weight.ln();
            sum_of_weights += weight;
            sum_of_weight_log_weights += weight_log_weights[t];
        }

        let mut model = OverlappingModel {
            wave: vec![false; t_count * width * height].into_boxed_slice(), // [[[false; T]; WIDTH]; HEIGHT],
            sums_of_ones: vec![0; width * height].into_boxed_slice(),       // [[0; WIDTH]; HEIGHT],
            entropies: vec![0.0; width * height].into_boxed_slice(), // [[0.0; WIDTH]; HEIGHT],
            rng: ChaCha20Rng::seed_from_u64(seed),
            observed: None,                      // [[0; WIDTH]; HEIGHT],
            weights: weights.into_boxed_slice(), // [0.0; T],
            compatible: vec![0; 4 * t_count * width * height].into_boxed_slice(), // [[[[0; 4]; T]; WIDTH]; HEIGHT],
            stack: Vec::with_capacity(t_count * width * height),
            sums_of_weights: vec![0.0; width * height].into_boxed_slice(), // [[0.0; WIDTH]; HEIGHT],
            sums_of_weights_log_weights: vec![0.0; width * height].into_boxed_slice(), // [[0.0; WIDTH]; HEIGHT],
            weight_log_weights: weight_log_weights.into_boxed_slice(),                 // [0.0; T],
            propagator: propagator.into_boxed_slice(), // [[&[usize]; T]; 4],
            big_n,
            periodic: periodic_output,
            sum_of_weights,
            sum_of_weight_log_weights,
            width,
            height,
            t_count,
            distribution: vec![0.0; t_count].into_boxed_slice(),
            starting_entropy: sum_of_weights.ln() - sum_of_weight_log_weights / sum_of_weights,
            patterns: patterns.into_boxed_slice(),
        };

        model.clear();
        model
    }

    pub fn observe(&mut self) -> Option<bool> {
        let mut min = 1e3;
        let mut argmin = None;

        for y in 0..self.height {
            for x in 0..self.width {
                if self.on_boundary((x as isize, y as isize)) {
                    continue;
                }

                let amount = self.sums_of_ones[y * self.width + x];
                if amount == 0 {
                    return Some(false);
                }

                let entropy = self.entropies[y * self.width + x];
                if amount > 1 && entropy <= min {
                    let noise = 1e-6 * self.rng.gen::<f64>();
                    if entropy + noise < min {
                        min = entropy + noise;
                        argmin = Some((x, y));
                    }
                }
            }
        }

        let argmin = match argmin {
            None => {
                let mut observed = vec![0; self.width * self.height];
                for y in 0..self.height {
                    for x in 0..self.width {
                        for t in 0..self.t_count {
                            if self.wave[(y * self.width + x) * self.t_count + t] {
                                observed[y * self.width + x] = t;
                                break;
                            }
                        }
                    }
                }
                self.observed = Some(observed.into_boxed_slice());
                return Some(true);
            }
            Some(i) => i,
        };

        for t in 0..self.t_count {
            self.distribution[t] =
                if self.wave[(argmin.1 * self.width + argmin.0) * self.t_count + t] {
                    self.weights[t]
                } else {
                    0.0
                }
        }

        let dist = WeightedIndex::new(self.distribution.iter()).unwrap();
        let rand = dist.sample(&mut self.rng);

        for t in 0..self.t_count {
            if self.wave[(argmin.1 * self.width + argmin.0) * self.t_count + t] != (t == rand) {
                self.ban(argmin, t);
            }
        }

        None
    }

    pub fn on_boundary(&self, (x, y): (isize, isize)) -> bool {
        !self.periodic
            && (x + self.big_n as isize > self.width as isize
                || y + self.big_n as isize > self.height as isize
                || x < 0
                || y < 0)
    }

    fn ban(&mut self, (x, y): (usize, usize), t: usize) {
        self.wave[(y * self.width + x) * self.t_count + t] = false;

        for d in 0..4 {
            self.compatible[((y * self.width + x) * self.t_count + t) * 4 + d] = 0;
        }

        self.stack.push(((x, y), t));

        self.sums_of_ones[y * self.width + x] -= 1;
        self.sums_of_weights[y * self.width + x] -= self.weights[t];
        self.sums_of_weights_log_weights[y * self.width + x] -= self.weight_log_weights[t];

        let sum = self.sums_of_weights[y * self.width + x];
        self.entropies[y * self.width + x] =
            sum.ln() - self.sums_of_weights_log_weights[y * self.width + x] / sum;
    }

    pub fn propagate(&mut self) {
        while let Some(((x, y), t)) = self.stack.pop() {
            for d in 0..4 {
                let (dx, dy) = (DX[d], DY[d]);
                let (mut x2, mut y2) = (x as isize + dx, y as isize + dy);

                if self.on_boundary((x2, y2)) {
                    continue;
                }

                if x2 < 0 {
                    x2 += self.width as isize;
                } else if x2 >= self.width as isize {
                    x2 -= self.width as isize;
                }
                if y2 < 0 {
                    y2 += self.height as isize;
                } else if y2 >= self.height as isize {
                    y2 -= self.height as isize;
                }

                let (x2, y2) = (x2 as usize, y2 as usize);

                for l in 0..self.propagator[t * 4 + d].len() {
                    let t2 = self.propagator[t * 4 + d][l];

                    self.compatible[((y2 * self.width + x2) * self.t_count + t2) * 4 + d] -= 1;

                    if self.compatible[((y2 * self.width + x2) * self.t_count + t2) * 4 + d] == 0 {
                        self.ban((x2, y2), t2);
                    }
                }
            }
        }
    }

    pub fn clear(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                for t in 0..self.t_count {
                    self.wave[(y * self.width + x) * self.t_count + t] = true;
                    for d in 0..4 {
                        self.compatible[((y * self.width + x) * self.t_count + t) * 4 + d] =
                            self.propagator[t * 4 + OPPOSITE[d]].len() as isize;
                    }
                }

                self.sums_of_ones[y * self.width + x] = self.weights.len() as isize;
                self.sums_of_weights[y * self.width + x] = self.sum_of_weights;
                self.sums_of_weights_log_weights[y * self.width + x] =
                    self.sum_of_weight_log_weights;
                self.entropies[y * self.width + x] = self.starting_entropy;
            }
        }
    }

    pub fn as_image(&self) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
        let mut buf = ImageBuffer::new(self.width as u32, self.height as u32);

        if let Some(ref observed) = self.observed {
            for y in 0..self.height {
                for x in 0..self.width {
                    let dy = if y < self.height - self.big_n + 1 {
                        0
                    } else {
                        (self.big_n - 1) as isize
                    };
                    let dx = if x < self.width - self.big_n + 1 {
                        0
                    } else {
                        (self.big_n - 1) as isize
                    };

                    let image::Rgba(inner) = self.patterns[observed
                        [(y as isize - dy) as usize * self.width + (x as isize - dx) as usize]]
                        .get_pixel(dx as u32, dy as u32);

                    buf.put_pixel(
                        x as u32,
                        y as u32,
                        image::Rgb([inner[0], inner[1], inner[2]]),
                    );
                }
            }
        } else {
            for y in 0..self.height {
                for x in 0..self.width {
                    let mut contributors = 0;
                    let (mut r, mut g, mut b) = (0u64, 0u64, 0u64);

                    for dy in 0..self.big_n {
                        for dx in 0..self.big_n {
                            let mut sx = x as isize - dx as isize;
                            if sx < 0 {
                                sx += self.width as isize;
                            }

                            let mut sy = y as isize - dy as isize;
                            if sy < 0 {
                                sy += self.height as isize;
                            }

                            if self.on_boundary((sx, sy)) {
                                continue;
                            }

                            for t in 0..self.t_count {
                                if self.wave
                                    [(sy as usize * self.width + sx as usize) * self.t_count + t]
                                {
                                    contributors += 1;

                                    let image::Rgba(inner) =
                                        self.patterns[t].get_pixel(dx as u32, dy as u32);

                                    r += inner[0] as u64;
                                    g += inner[1] as u64;
                                    b += inner[2] as u64;
                                }
                            }
                        }
                    }

                    if contributors > 0 {
                        buf.put_pixel(
                            x as u32,
                            y as u32,
                            image::Rgb([
                                (r / contributors) as u8,
                                (g / contributors) as u8,
                                (b / contributors) as u8,
                            ]),
                        );
                    }
                }
            }
        }

        buf
    }
}

fn sample_image<I: GenericImageView>(
    img: &I,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    in_w: usize,
    in_h: usize,
) -> ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>
where
    I::Pixel: Debug + 'static,
{
    let mut buf = ImageBuffer::new(w as u32, h as u32);
    for dy in 0..h {
        for dx in 0..w {
            let pixel = img.get_pixel(((x + dx) % in_w) as u32, ((y + dy) % in_h) as u32);
            buf.put_pixel(dx as u32, dy as u32, pixel);
        }
    }
    buf
}

fn is_neighbor<I: GenericImageView>(p1: &I, p2: &I, dx: isize, dy: isize) -> bool {
    let mut xmin = dx;
    let mut xmax = p1.width() as isize;
    let mut ymin = dy;
    let mut ymax = p1.height() as isize;

    if dx < 0 {
        xmin = 0;
        xmax = dx + p1.width() as isize;
    }

    if dy < 0 {
        ymin = 0;
        ymax = dy + p1.height() as isize;
    }

    for y in ymin..ymax {
        for x in xmin..xmax {
            if p1.get_pixel(x as u32, y as u32).to_rgb()
                != p2.get_pixel((x - dx) as u32, (y - dy) as u32).to_rgb()
            {
                return false;
            }
        }
    }
    true
}

fn main() {
    let img = image::open("./dead look.png").unwrap();

    for tr in 0.. {
        let mut model =
            OverlappingModel::new(&img, 3, 128, 128, false, false, Symmetries::None, random());

        let mut i = 0;
        let res = loop {
            println!("[{}] Turn {}", tr, i);
            // println!("{:?}", model);

            if let Some(res) = model.observe() {
                break res;
            }

            model.propagate();

            // if i % 100 == 0 {
                model.as_image().save(format!("./output/output{}.png", i)).unwrap();
            // }
            i += 1;
        };

        model.as_image().save(format!("./output/output{}.png", i)).unwrap();

        println!(
            "Result: {}",
            match res {
                true => "success!",
                false => "contradiction!",
            }
        );

        if res {
            break;
        }
    }
}
