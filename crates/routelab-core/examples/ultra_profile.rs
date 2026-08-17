//! Where ULTRA's preprocessing time goes.
//!
//! A throwaway harness, not part of the library's surface. It builds a
//! transit-shaped timetable over a connected transfer graph and runs
//! [`Ultra::compute`] on it, with one knob that matters: `pad`, which widens
//! the *vertex numbering* without adding a single reachable vertex.
//!
//! That knob is the experiment. A multimodal environment numbers every street
//! corner — 560,706 on Seattle — while Core-CH leaves a core of ~17,000 that
//! is all the preprocessing ever touches. The label arrays are sized by the
//! numbering rather than by the core, so the same work is spread over 32× the
//! address space. Running the same network at pad=1 and pad=32 says what that
//! costs, with the shortcuts identical either way.
//!
//! ```text
//! cargo run --profile profiling --example ultra_profile -- 1
//! cargo run --profile profiling --example ultra_profile -- 32
//! ```

use std::time::Instant;

use routelab_core::{Connection, Graph, NodeId, Timetable, TripId, Ultra};

/// A small LCG; the library's own is crate-private and this is a harness.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }
    fn below(&mut self, bound: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % bound.max(1)
    }
}

/// How many stops, lines, and trips to build. Roughly King County Metro's
/// shape at a fraction of its size, so a run finishes while you watch.
struct Shape {
    /// Vertices in the transfer graph, all mutually reachable — the *core*,
    /// what Core-CH leaves standing. 17,454 on Seattle.
    core: u32,
    /// How many of those a vehicle calls at, spread through the core. 6,313
    /// on King County Metro.
    stops: u32,
    lines: u32,
    line_len: u32,
    trips: u32,
    /// Multiplies the vertex numbering without adding reachable vertices.
    /// Seattle numbers 560,706 for a core of 17,454, so roughly 32.
    pad: u32,
}

impl Shape {
    /// Which vertex the `n`th served stop is, spread through the core.
    fn stop(&self, n: u32) -> NodeId {
        (n % self.stops) * (self.core / self.stops)
    }
}

/// A transit-shaped network: lines that run a contiguous stretch of stops
/// several times a day, over a transfer graph that is connected, so that
/// unlimited walking reaches everything the way a street network does.
fn network(seed: u64, shape: &Shape) -> (Timetable, Graph) {
    let mut rng = Rng::new(seed ^ 0x0117a);
    let mut connections = Vec::new();
    let mut trip = 0u32;

    for _ in 0..shape.lines {
        // A line runs a stretch of consecutive served stops, so riders on it
        // share stops with a few other lines rather than with all of them.
        let first = rng.below(u64::from(shape.stops - shape.line_len)) as u32;
        for _ in 0..shape.trips {
            // Spread over a service day, so a stop has many distinct
            // departure times — which is what ULTRA sweeps over.
            let mut now = rng.below(18 * 3600) as u32;
            for i in 0..shape.line_len - 1 {
                let departs = now + rng.below(60) as u32;
                let arrives = departs + 45 + rng.below(90) as u32;
                connections.push(Connection {
                    trip: TripId(trip),
                    from: shape.stop(first + i),
                    to: shape.stop(first + i + 1),
                    departs,
                    arrives,
                });
                now = arrives;
            }
            trip += 1;
        }
    }

    // The transfer graph: short hops to nearby stops, both ways, connected end
    // to end. Unlimited walking therefore reaches every stop, which is the
    // case that makes the preprocessing expensive.
    let mut links = Vec::new();
    for v in 0..shape.core {
        for step in 1..=2u32 {
            let next = (v + step) % shape.core;
            let walk = 60 + rng.below(180) as u32;
            links.push((v, next, walk));
            links.push((next, v, walk));
        }
    }

    // The padding: vertices numbered but joined to nothing, so the answer is
    // identical and only the address space grows.
    let vertices = (shape.core * shape.pad) as usize;
    (
        Timetable::new(vertices, connections),
        Graph::from_edges(vertices, &links).expect("links drawn from the core"),
    )
}

fn main() {
    let pad: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1);
    // Seattle's proportions: a core of ~17.5k reachable vertices, ~6.3k of
    // them served, numbered ~32x wider than the core. Fewer trips than a real
    // service day so a run finishes while you watch — that scales the source
    // sweep linearly and leaves the per-search cache regime, which is what
    // this is measuring, exactly where the real thing puts it.
    let shape = Shape {
        core: 17_500,
        stops: 6_300,
        lines: 300,
        line_len: 20,
        trips: 3,
        pad,
    };

    let built = Instant::now();
    let (table, transfers) = network(7, &shape);
    let departures: usize = table.connections().len();
    println!(
        "network: {} served of {} core, numbered {} (pad {}), {} connections, {} transfer arcs, built in {:.1?}",
        shape.stops,
        shape.core,
        transfers.num_nodes(),
        pad,
        departures,
        transfers.num_edges(),
        built.elapsed()
    );

    let run = Instant::now();
    let ultra = Ultra::compute(&table, &transfers);
    let elapsed = run.elapsed();
    println!(
        "ULTRA: {:.2?}  ({} shortcuts from {} candidates)",
        elapsed,
        ultra.len(),
        ultra.candidates()
    );
}
