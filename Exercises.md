# Exercises

## Week of July 24th

### 4.1

q(11, down) = -1
q(7, down) = -1 + v(11) = -15

### 4.2

q(15) = -20 in either case because the states transitioned to from new state 15 have identical
v in all cases to the original MDP

### 4.3

See paper!

### 4.5

See `car_rental.py`

### 5.1

Jump at the back is due to significant probability of dealer going bust of sticking on 17-19,
without having ourselves gone bust (we're conditioned on that already in these states and
most of this is the conditioning on us not having gone bust).

Drop off to left is due to the dealer's added flexibility from a usable ace which effectively
gives him a retry-on-bust

### 5.2

Exactly the same since in this game it is not possible to repeat a state within an episode

### 5.4

```Returns(s, a) <- empty list```

becomes

```
mean_return(s, a) <- 0
count)(s, a) <- 0
```

and

```
Append G to Returns(s, a)
Q(St, At) <- avg(Returns(St, At))
```
becomes

```
mean_return(s, a) <- (mean_return(s, a)*count(s, a) + G)/(count(s, a) + 1)`
count(s, a) <- count(s, a) + 1
Q(St, At) <- mean_return(St, At)
```

### 5.12

See `racetrack.py`