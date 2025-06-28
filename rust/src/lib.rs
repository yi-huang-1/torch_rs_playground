use pyo3::prelude::*;
use std::collections::HashMap;
use pyo3::IntoPyObjectExt;
use pyo3_tch::{wrap_tch_err, PyTensor};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn hello() -> PyResult<String> {
    println!("Rust says: Hello, World!");
    Ok("Hello, World!".to_string())
}

// Fibonacci
#[pyfunction]
fn fibonacci(n: u64) -> u64 {
    if n <= 0 {
        panic!("n must be greater than 0");
    }

    match n {
        0|1|2 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

// Fibonacci Number Map
#[pyfunction]
fn fibonacci_number_map(numbers: Vec<u64>) -> PyResult<Py<PyAny>> {
    let mut n_map = HashMap::new();
    for &n in numbers.iter() {
        let count = n_map.entry(n.to_string()).or_insert(0);
        *count = fibonacci(n);
    }
    Python::with_gil(|py| n_map.into_py_any(py))
}


// torch tensor add_one
#[pyfunction]
fn torch_tensor_add_one(tensor: PyTensor) -> PyResult<PyTensor> {
    let tensor = tensor.f_add_scalar(1.0).map_err(wrap_tch_err)?;
    Ok(PyTensor(tensor))
}

// torch tensor multiply
#[pyfunction]
fn torch_tensor_multiply(tensor: PyTensor, other: PyTensor) -> PyResult<PyTensor> {
    let tensor = tensor.f_mul(&other).map_err(wrap_tch_err)?;
    Ok(PyTensor(tensor))
}


/// A Python module implemented in Rust.
#[pymodule]
fn torchrdit_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(fibonacci, m)?)?;
    m.add_function(wrap_pyfunction!(fibonacci_number_map, m)?)?;
    m.add_function(wrap_pyfunction!(torch_tensor_add_one, m)?)?;
    m.add_function(wrap_pyfunction!(torch_tensor_multiply, m)?)?;
    Ok(())
}
