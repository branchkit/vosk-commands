fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-search=native={manifest_dir}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{manifest_dir}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
}
