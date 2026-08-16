//! Routing over a timetable: the same question asked of two different graphs.
//!
//! Pyrga, Schulz, Wagner & Zaroliagis, *Efficient Models for Timetable
//! Information in Public Transportation Systems* (ACM Journal of Experimental
//! Algorithmics 12, Article 2.4, 2007). A timetable is not a network with
//! weights on it — it is a set of **connections**, each one a vehicle leaving
//! one stop at one instant and reaching another at another. Turning that into
//! something a shortest-path algorithm can read is the whole problem, and the
//! paper gives two answers:
//!
//! - **Time-expanded** ([`TimeExpanded`]) — a node per *event*. Every departure
//!   and every arrival is its own node, connections and waiting are edges, and
//!   what comes out is an ordinary static graph that [`crate::dijkstra`] routes
//!   with no changes at all.
//! - **Time-dependent** ([`earliest_arrival`]) — a node per *stop*. Far fewer
//!   nodes, but each edge carries the connections running along it and
//!   traversing one means finding the next departure, so the search has to be
//!   written for it.
//!
//! Both read the same optional third thing: [`crate::model::timetable::Footpaths`], the paper's
//! *foot-edges* — a link between two stops a rider can take at any time for a
//! fixed duration, which is what lets a real feed's northbound and southbound
//! stops across a street be the same place for the purpose of changing buses.
//! Without them a city timetable is routable only between stops a single trip
//! chain connects, and a query dropped at the wrong side of the street has no
//! answer.
//!
//! Both answer with the same verb — `earliest_arrival` — because comparing them
//! is the point. The two must agree on every query; that is the paper's thesis
//! and it is also this module's main test, since neither model is the reference
//! implementation. Each is the other's.

mod dependent;
mod expanded;

#[cfg(test)]
mod tests;

pub use dependent::earliest_arrival;
pub use expanded::TimeExpanded;
