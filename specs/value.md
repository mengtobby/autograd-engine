# Value

## Purpose
A single scalar node in the computation graph. It holds a number from the
forward pass and collects that number's gradient during the backward pass.
Every op (specs/add.md, specs/ops.md) takes Values and returns a new Value.

## Interface
`Value(data, _prev=(), _op="")`

## Fields
| field | what it holds |
|---|---|
| `data` | The number this node computed in the forward pass (a float, not just ints). |
| `grad` | ∂L/∂(this node): how much the *final loss* L changes per unit change in `data`. Starts at 0 and is built up with `+=` during backward. Not the local gradient: local gradients are computed on the fly inside `_backward` and never stored. |
| `_prev` | The node's parents: the Values it was made from. For `e = a * b`, `e._prev = (a, b)`; for `d = e + c`, `d._prev = (e, c)`. A node only knows its direct parents; following these links back from L reaches the whole graph. |
| `_backward` | A function attached by the op that created this node, holding that op's backward rule (e.g. the multiply rule). `backward()` calls `node._backward()` without knowing which op it is. Chosen over an op-name + `if/elif` design so each op's forward and backward live together, and adding a new op touches one place only. |
| `_op` | Label for the op that made this node (`"+"`, `"*"`, `"tanh"`, ...); `""` for leaves. Only for printing and drawing the graph with graphviz; backward never reads it. |

## Edge cases
- **Leaf nodes** (created directly, e.g. `a = Value(2.0)`, not by an op): `_prev = ()` and `_backward` is an empty function. A leaf's `_backward` does nothing since it has no parents. Its gradient stays in `.grad`, where the training loop reads it. Giving leaves a do-nothing `_backward` means `backward()` can call it on every node with no special case.

## Tests I want
- `Value(2.0)` has `data == 2.0`, `grad == 0`, no parents, `_op == ""`, and calling its `_backward()` changes nothing.
- For `e = a * b`: `e._prev` contains exactly `a` and `b`, and `e._op == "*"`.
