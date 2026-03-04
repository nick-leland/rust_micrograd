extern crate rand;

use rand::Rng;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

#[derive(Default)]
struct Value {
    value: f64,
    grad: f64,
    children: Vec<Node>,
    operation: String,
    label: String,
}

impl Value {
    fn no_children(&self) -> bool {
        self.children.is_empty()
    }
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = " ".repeat(indent);

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

    fn backward(&self) {
        // Assume self.grad is already set to the gradient of the output node
        let op = self.operation.clone();
        let out_grad = self.grad;

        // Based on operation, compute the gradient of the children
        match op.as_str() {
            "+" => {
                self.children[0].set_grad(self.children[0].grad() + out_grad);
                self.children[1].set_grad(self.children[1].grad() + out_grad);
            }
            "-" => {
                self.children[0].set_grad(self.children[0].grad() + out_grad);
                self.children[1].set_grad(self.children[1].grad() + -out_grad);
            }
            "*" => {
                self.children[0]
                    .set_grad(self.children[0].grad() + (self.children[1].value() * out_grad));
                self.children[1]
                    .set_grad(self.children[1].grad() + (self.children[0].value() * out_grad));
            }
            "/" => {
                self.children[0]
                    .set_grad(self.children[0].grad() + (out_grad / self.children[1].value()));
                self.children[1].set_grad(
                    self.children[1].grad()
                        + ((-out_grad * self.children[0].value())
                            / (self.children[1].value() * self.children[1].value())),
                );
            }
            _ => {}
        }

        // Now we perform the DFS
        if self.no_children() {
        } else {
            for leaf in self.children.iter() {
                leaf.backward();
            }
        }
    }
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

    fn with_op(
        value: f64,
        label: impl Into<String>,
        op: impl Into<String>,
        children: Vec<Node>,
    ) -> Self {
        Node(Rc::new(RefCell::new(Value {
            value,
            grad: 0.0,
            children,
            operation: op.into(),
            label: label.into(),
        })))
    }

    fn add(&self, other: &Node, label: impl Into<String>) -> Node {
        Node::with_op(
            self.value() + other.value(),
            label,
            "+",
            vec![self.clone(), other.clone()],
        )
    }

    fn subtract(&self, other: &Node, label: impl Into<String>) -> Node {
        Node::with_op(
            self.value() - other.value(),
            label,
            "-",
            vec![self.clone(), other.clone()],
        )
    }

    fn mul(&self, other: &Node, label: impl Into<String>) -> Node {
        Node::with_op(
            self.value() * other.value(),
            label,
            "*",
            vec![self.clone(), other.clone()],
        )
    }

    fn div(&self, other: &Node, label: impl Into<String>) -> Node {
        Node::with_op(
            self.value() / other.value(),
            label,
            "/",
            vec![self.clone(), other.clone()],
        )
    }

    fn value(&self) -> f64 {
        self.0.borrow().value
    }

    fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad
    }

    fn backward(&self) {
        self.0.borrow().backward()
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

    // a = 2.0
    // b = -3.0
    // c = 10.0
    // e = a * b
    // d = e + c
    // f = -2.0
    // L = d * f

    {
        println!("New Operations");
        let a = Node::new(2.0, "a");
        let b = Node::new(-3.0, "b");
        let c = Node::new(10.0, "c");
        let f = Node::new(-2.0, "f");

        let e = a.mul(&b, "e");
        let d = e.add(&c, "d");
        let l = d.mul(&f, "L");

        println!("{}", l);
        l.set_grad(1.0);
        l.backward();
        println!("\n\n");
        println!("{}", l);
    }
}
