extern crate serde;

use serde_json::{Result, Value};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Data {
    selftext: str,
}

fn main() -> Result<()> {
    let stdin = std::io::stdin();
    let stdin = stdin.lock();

    let deserializer = serde_json::Deserializer::from_reader(stdin);
    let iterator = deserializer.into_iter::<serde_json::Value>();
    for item in iterator {
        println!("Got {:?}", item?);
    }

    Ok(())

}
