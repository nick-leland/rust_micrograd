extern crate rand;

use rand::Rng;
use std::fmt;

struct Value {
    value: f64,
    grad: f64,
    children: Vec<Value>,
    operation: String,
    label: String,
    // Maybe add a dictionary for values that are computed in relation?
}


impl Value {
    fn has_children(&self) -> bool {
        self.children.is_empty()
    }
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = " ".repeat(indent);

        // General Definition
        // TODO Should only print grad if there is a grad value
        write!(
            f,
            "{}{}={}, grad={}",
            pad, self.label, self.value, self.grad
        )?;

        if self.children.is_empty() {
            return Ok(());
        }

        write!(f, "{} children:", pad)?;

        for (i, child) in self.children.iter().enumerate() {
            write!(f, "\n{}[{}] ", pad, i)?;
            child.fmt_with_indent(f, indent + 1)?;
        }

        Ok(())
    }
    // Have to calculate the gradient with respect to the target point
    // We can call this recursively, then apply the chain rule back to the loss
    // a = 2.0
    // b = -3.0
    // c = 10.0
    // e = a * b
    // d = e + c
    // f = -2.0
    // L = d * f

    fn backprop(&self, eval: String) {
        // Derivative of all children with respect to self
        for (child) in self.children {
            // Update gradient with derivative based on operation
            // +
            // -
            // *
            // /
        }
    } 
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_indent(f, 0)
    }
}

fn main() {
    // Imagine we are initializing wights for a NN
    println!("Testing Random");
    let weight_a = rand::thread_rng().next_f64();
    let bias_a = rand::thread_rng().next_f64();
    println!("Initialized weight_a | {}", weight_a);
    println!("Initialized bias_a | {}", bias_a);

    // Let's look at a basic equation now
    let a: Value = Value {
        value: 2.0,
        grad: 0.0,
        children: Vec::new(),
        operation: String::new(),
        label: String::from("a"),
    };

    let b: Value = Value {
        value: -3.0,
        grad: 0.0,
        children: Vec::new(),
        operation: String::new(),
        label: String::from("b"),
    };
    let c: Value = Value {
        value: 10.0,
        grad: 0.0,
        children: Vec::new(),
        operation: String::new(),
        label: String::from("c"),
    };

    let e: Value = Value {
        value: a.value * b.value,
        grad: 0.0,
        children: vec![a, b], // vec! parameter is a macro to allow for hodling of anytype
        operation: String::from("*"),
        label: String::from("e"),
    };

    let d: Value = Value {
        value: e.value + c.value,
        grad: 0.0,
        children: vec![e, c], // vec! parameter is a macro to allow for hodling of anytype
        operation: String::from("*"),
        label: String::from("d"),
    };

    let f: Value = Value {
        value: -2.0,
        grad: 0.0,
        children: Vec::new(), // vec! parameter is a macro to allow for hodling of anytype
        operation: String::new(),
        label: String::from("f"),
    };

    let L: Value = Value {
        value: d.value * f.value,
        grad: 0.0,
        children: vec![d, f], // vec! parameter is a macro to allow for hodling of anytype
        operation: String::from("*"),
        label: String::from("L"),
    };

    println!("{}", L);
}
