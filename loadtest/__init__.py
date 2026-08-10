"""Load generation and comparison plots for the five rate limiting algorithms.

Not part of the shipped package: this is the measurement rig, not the
library. It exists to answer one question with real data rather than
argument -- given *identical* traffic, how differently do the five
algorithms behave, and where exactly does that difference show up?

The short answer, which the plots make visible, is that the differences are
almost entirely at window boundaries and under bursts. Under smooth traffic
well below the limit, all five are indistinguishable.
"""
