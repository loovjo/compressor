use std::env::args;
use std::io::{Read, Write, Result, stdin, stdout};
use std::fs::File;

mod compress;

fn main() {
    if let Err(e) = start() {
        eprintln!("Error: {:?}", e);
    }
}

fn start() -> Result<()> {
    if let Some(x) = args().nth(1) {
        match x.as_ref() {
            "-e" => {
                let data = read_input()?;
                let compressed = compress::compress(data.as_ref());

                write_output(compressed.as_slice())?;
            }
            "-d" => {
                let data = read_input()?;
                if let Some(decompressed) = compress::decompress(data) {
                    write_output(decompressed.as_slice())?;
                } else {
                    eprintln!("Bad input");
                }
            }
            _ => {
                eprintln!("Usage: -[ed] [input] [output]");
            }
        }
    } else {
        eprintln!("Usage: -[ed]");
    }
    Ok(())
}

fn read_input() -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    if let Some(x) = args().nth(2) {
        let mut file = File::open(x)?;
        file.read_to_end(&mut buf)?;
    } else {
        stdin().read_to_end(&mut buf)?;
    }
    Ok(buf)
}

fn write_output(data: &[u8]) -> Result<()> {
    if let Some(x) = args().nth(3) {
        let mut file = File::create(x)?;
        file.write_all(data)?;
    } else {
        stdout().write_all(data)?;
    }
    Ok(())
}
