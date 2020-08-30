extern crate ajson;
extern crate csv;
extern crate regex;
#[macro_use] extern crate lazy_static;

use std::path::Path;
use std::fs;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use csv::Writer;
use regex::Regex;

/*
fn trawl_files() -> Result<Vec<String>, Box<dyn Error>> {
    // want to look through a directory, and get a list of 
    // .json files  
    let mut files = read_dir("./data/")?
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, io::Error>>()?;
    Ok(files)
}
*/

fn matching(v: &str) -> Result<String, Box<dyn Error>> { 
    lazy_static! {
        static ref RE: Regex = Regex::new(r"(tl;dr|tl:dr).*").unwrap();
    }
    let caps = RE.captures(v).unwrap();
    Ok(caps[0].to_owned())
}   

fn parse_json(p: &Path) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"(tl;dr|tl:dr).*").unwrap();
    }
    let mut vals = Vec::new();
    for entry in fs::read_dir(p)? {
        let entry = entry?;
        let path = entry.path();
        let f = File::open(path)?;
        let reader = BufReader::new(f);
        for item in reader.lines() {
            let txt = ajson::get(&item.unwrap(), "selftext");
            let txt = txt.unwrap();
            let lower = txt.as_str().to_lowercase();
            if lower.contains("tl;dr") | lower.contains("tl:dr") {
                let caps = RE.captures(&lower).unwrap();
                let rep = RE.replace_all(&lower, "").into();
                vals.push((rep, caps[0].to_owned()));
            }
        }
    }
    Ok(vals)
}

fn csv_conv(v: Vec<(String, String)>) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path("res.csv")?;
    wtr.write_record(&["Text", "Summary"])?;
    for item in v {
        wtr.write_record(&[item.0, item.1])?;
    }
    wtr.flush()?;
    Ok(())
}

fn main() {
    let dir = Path::new(&"./src/data/");
    let vals = parse_json(dir).unwrap(); 
    csv_conv(vals);
}
