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
