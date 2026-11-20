# Notes

Predictions, results, surprises. One entry per piece.

Entry format:

```
## <date> — <piece>
Prediction:
Result:
Surprise / what I got wrong:
One-line takeaway:
```

---

## 2026-11-20 — hand trace (Day 1, Block 3)
a = 2, b = -3, c = 10; e = a*b, d = e + c, L = d*d
Prediction: forward e = -6, d = 4, L = 16.
Backward L.grad = 1, d.grad = 8, e.grad = 8, c.grad = 8, a.grad = -24, b.grad = 16.
Result: (Block 5: compare with engine and PyTorch)
Surprise / what I got wrong: twice did `4 += 4` as 16. `+=` adds the new
piece to what's there (4 + 4 = 8). Wasn't sure how add passes gradient: it
passes it through unchanged to every input.
One-line takeaway: `d * d` uses d twice, so d collects 4 + 4 = 8 = 2d.

## 2026-11-20 — backward: break it predictions (Day 1, Block 6)

**Don't set the output grad to 1**
Prediction: every gradient comes out 0. Each backward step multiplies by the
grad of the node after it, so a 0 seed passes 0 down the whole chain.
Weights never change, so nothing learns, and nothing crashes (silent).
Result:
Surprise / what I got wrong:
One-line takeaway:

**`=` instead of `+=`**
Prediction: any value used more than once gets only its last use's gradient.
`a * a` at a = 3 gives a.grad = 3 instead of 6. Tests that catch it: `a * a`,
`a + a`. No crash, just a wrong number.
Result:
Surprise / what I got wrong:
One-line takeaway:
