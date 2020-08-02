extern crate ajson;
extern crate csv;

use std::path::Path;
use std::error::Error;
use std::fs::{File, read_dir};
use std::io::{BufRead, BufReader};
use csv::Writer;
use std::io;

fn trawl_files() -> Result<Vec<String>, Box<dyn Error>> {
    // want to look through a directory, and get a list of 
    // .json files  
    let mut files = read_dir("./data/")?
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, io::Error>>()?;
    Ok(files)
}


fn parse_json(p: String) -> Result<Vec<String>, Box<dyn Error>> {
    let mut vals = Vec::new();
    let f = File::open(p.to_string()).unwrap();
    let reader = BufReader::new(f);

    for item in reader.lines() {
        let txt = ajson::get(&item.unwrap(), "selftext");
        let txt = txt.unwrap();
        let lower = txt.as_str().to_lowercase();
        if lower.contains("tl;dr") | lower.contains("tl:dr") {
            vals.push(lower);
        }
    }
    Ok(vals)
}

fn csv_conv(V: Vec<String>) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path("res.csv")?;
    wtr.write_record(V)?;
    wtr.flush()?;
    Ok(())
}

fn main() {
    let mut files = trawl_files().unwrap();
    println!("{:?}", files)
    let mut vals = parse_json("./src/data/2011-01.json".to_string());
    let vals = vals.unwrap();
    println!("{}", vals.len());
    csv_conv(vals);
}
