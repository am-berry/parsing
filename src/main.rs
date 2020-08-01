extern crate ajson;
extern crate csv;

use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use csv::Writer;

fn parse_json(p: String) -> Result<Vec<String>, Box<dyn Error>> {
    let mut vals = Vec::new();
    let f = File::open(p.to_string()).unwrap();
    let reader = BufReader::new(f);

    for item in reader.lines() {
        let txt = ajson::get(&item.unwrap(), "selftext");
        let txt = txt.unwrap();
        let lower = txt.as_str().to_lowercase();
        if lower.contains("tl;dr") | lower.contains("tl:dr") {
            println!("{:?}", lower);
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
    let mut vals = parse_json("./src/data/2011-01.json".to_string());
    csv_conv(vals.unwrap());
}
