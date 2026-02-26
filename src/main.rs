extern crate rand;

use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;
use std::{fmt, rc};

#[derive(Default)]
struct Value {
    value: f64,
    grad: f64,
    // children: Vec<Value>, // Argument that this could be a HashSet?
    children: Vec<Node>, // TODO IMPLIMENT THIS (allow for node values nested
    // generation and same value in expression)
    operation: String,
    label: String,
    // visited: Vec<Value>,
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
    fn _dfs(&self) {}
    fn _backprop(&self) {}
    // We are going to make some slight changes to the algorithm below (because we aren't
    // searching for a key)
    // if x == null or k == x.key
    //  return x
    // if k < x.key
    //  return DFS(left)
    // else return DFS(right)
}

#[derive(Clone)]
struct Node(Rc<RefCell<Value>>);

impl Node {
    fn new(value: f64, label: impl Into<String>) -> Self {
        Node(Rc::new(RefCell::new(Value {
            value,
            grad: 0.0,
            children: vec![],
            operation: "".into(),
            label: label.into(),
        })))
    }
    fn value(&self) -> f64 {
        self.0.borrow().value
    }
    fn label(&self) -> String {
        self.0.borrow().label.clone()
    }
    // HERE IS WHERE I AM RIGHT NOW
    // I think I can copy this?
    fn has_children(&self) -> bool {
        self.0.borrow().children.is_empty()
    }
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        self.0.borrow().fmt_with_indent(f, indent)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_indent(f, 0)
    }
}

impl fmt::Display for Node {
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
        label: String::from("a"),
        ..Default::default()
    };

    let b: Value = Value {
        value: -3.0,
        label: String::from("b"),
        ..Default::default()
    };

    let c: Value = Value {
        value: 10.0,
        label: String::from("c"),
        ..Default::default()
    };

    let e: Value = Value {
        value: a.value * b.value,
        children: vec![Node::new(a.value, a.label), Node::new(b.value, b.label)], // vec! parameter is a macro to allow for hodling of anytype
        operation: String::from("*"),
        label: String::from("e"),
        ..Default::default()
    };

    let d: Value = Value {
        value: e.value + c.value,
        // children: vec![e, c], // vec! parameter is a macro to allow for hodling of anytype
        children: vec![Node::new(e.value, e.label), Node::new(c.value, c.label)], // vec! parameter is a macro to allow for hodling of anytype
        operation: String::from("*"),
        label: String::from("d"),
        ..Default::default()
    };

    let f: Value = Value {
        value: -2.0,
        label: String::from("f"),
        ..Default::default()
    };

    let l: Value = Value {
        value: d.value * f.value,
        children: vec![Node::new(d.value, d.label), Node::new(f.value, f.label)], // vec! parameter is a macro to allow for hodling of anytype
        operation: String::from("*"),
        label: String::from("L"),
        ..Default::default()
    };

    println!("{}, {}", l, l.operation);
}
