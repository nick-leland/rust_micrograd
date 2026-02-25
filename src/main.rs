extern crate rand;

use rand::Rng;
use std::fmt;

struct Value {
    value: f64,
    grad: f64,
    children: Vec<Value>,
    operation: String,
    label: String,
}

// This is the initial attempt that I had at providing the display
// The real pain point here was having to manually recall each node which was just a Value object
// instead of applying any sort of recursion
// impl Value {
//     fn has_children(&self) -> bool {
//         self.children.is_empty()
//     }
// }
//
// impl fmt::Display for Value {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         let has_children = self.has_children();
//         let applied_gradient = self.grad == 1.0;
//
//         if !applied_gradient && has_children {
//             write!(
//                 f,
//                 "Value {}: {} | Grad: {} | Children of {}: [",
//                 self.label, self.value, self.grad, self.label
//             )?;
//             for (i, v) in self.children.iter().enumerate() {
//                 if v.grad == 1.0 {
//                     write!(
//                         f,
//                         "  Child {} | {}: {} | Grad: {}",
//                         i, v.label, v.value, v.grad
//                     )?;
//                 } else {
//                     write!(f, "  Child {} | {}: {}", i, v.label, v.value)?;
//                 }
//             }
//             write!(f, "]")
//         } else if !applied_gradient {
//             write!(f, "Case B, test {}", self.value)
//         } else if has_children {
//             write!(f, "Case C, test {}", self.value)
//         } else {
//             write!(f, "Case D, test {}", self.value)
//         }
//
//     }
// }

impl Value {
    fn has_children(&self) -> bool {
        self.children.is_empty()
    }
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = " ".repeat(indent);

        // General Definition
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
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_indent(f, 0)
    }
}

fn main() {
    println!("Hello, world!");

    // Imagine we are initializing wights for a NN
    let weight_a = rand::thread_rng().next_f64();
    let bias_a = rand::thread_rng().next_f64();
    println!("Initialized weight_a | {}", weight_a);
    println!("Initialized bias_a | {}", bias_a);

    // Let's look at a basic equation now
    let a: Value = Value {
        value: 2.0,
        grad: 1.0,
        children: Vec::new(),
        operation: String::new(),
        label: String::from("a"),
    };

    println!("Has Children Result {}", a.has_children());

    let b: Value = Value {
        value: -3.0,
        grad: 1.0,
        children: Vec::new(),
        operation: String::new(),
        label: String::from("b"),
    };
    let c: Value = Value {
        value: 10.0,
        grad: 1.0,
        children: Vec::new(),
        operation: String::new(),
        label: String::from("c"),
    };

    let e: Value = Value {
        value: a.value * b.value,
        grad: 1.0,
        children: vec![a, b], // vec! parameter is a macro to allow for hodling of anytype
        operation: String::from("*"),
        label: String::from("e"),
    };

    let d: Value = Value {
        value: e.value + c.value,
        grad: 1.0,
        children: vec![e, c], // vec! parameter is a macro to allow for hodling of anytype
        operation: String::from("*"),
        label: String::from("d"),
    };

    // println!("Result of d: {}", d.value);

    println!("{}", d);
    // println!("Children of d: {}", d.children);

    // {
    //     // Input Equation: 3a^2 + 4b + c
    //     let d = 3.0 * f64::powf(a.value, 2.0) + 4.0 * b.value + c.value;
    //     println!("Result of D1: {}", d);
    // }

    // {
    //     // We can also represent this in a function
    //     fn equation(input_1: f64, input_2: f64, input_3: f64) -> f64 {
    //         3.0 * f64::powf(input_1, 2.0) + 4.0 * input_2 + input_3
    //     }

    //     let d: f64 = equation(a.value, b.value, c.value);
    //     println!("Result of D2: {}", d);
    // }
}
