extern crate serde;

use serde_json::{Result, Value};
use serde::{Serialize, Deserialize};

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Deserialize, Debug)]
struct Data {
    selftext: String,
}

fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Data, Box<Error>> {
    let stdin = std::io::stdin();
    let stdin = stdin.lock();

    let deserializer = serde_json::Deserializer::from_reader("./data/2011-01.json");
    let iterator = deserializer.into_iter::<serde_json::Value>();
    for item in iterator {
        println!("Got {:?}", item?);
    }

    Ok(())

}
