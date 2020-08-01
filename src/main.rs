extern crate ajson;

use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let mut vals = Vec::new();
    let f = File::open("src/data/2011-01.json").unwrap();
    let reader = BufReader::new(f);

    for item in reader.lines() {
        let txt = ajson::get(&item.unwrap(), "selftext").unwrap().as_str().to_lowercase();
        if txt.contains("tl;dr") | txt.contains("tl:dr") {
            vals.push(txt);
        }
    } 
    println!("{:?}", vals);
}

