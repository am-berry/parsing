extern crate serde;

use serde_json::{Result, Value};
use serde::{Deserialize};

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Deserialize, Debug)]
struct Data {
    selftext: String,
}

fn read_from_file<P: AsRef<Path>>(path: P) -> Result<()> {
    let file = File::open(path);
    let reader = BufReader::new(file);
    
    let deserializer = serde_json::Deserializer::from_reader(reader);
    let iterator = deserializer.into_iter::<Value>();
    for item in iterator {
        let v: Value = item["selftext"];
    Ok(())
}

fn main() {
   read_from_file("./data/2011-01.json").unwrap();
}
