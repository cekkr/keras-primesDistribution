# Prime numbers distribution prediction
This is a project for creating an algorithm using a Q-Learning model for predicting the distribution of prime numbers using most simple operations as possible.

Born as colab file: https://colab.research.google.com/drive/1hiBg7hLClOtAwb66UpPGUQcUA7I2NP3S

## Incipit
The entire project is based on a simple probability formula:
$notPrimePredictProb = (\frac{1}{n}*(1-notPrimePredictProb))+notPrimePredictProb$

The concept is: you can obtain the statistic prediction of not prime numbers and, by consequence, of prime numbers updating this formula for each prime number. This probability is valid until the next prime number found, so the intent of this project is finding the right algorithm which can use this simple concept for forecast with more precision as possible the prime numbers distribution from 2 to *n*.

Were tested many combinations of algorithms which, using this base formula, can obtain a pretty plausible probability distribution, although not able to predict the right prime numbers in the long term of the cycles.

These promising results led me to work to this Tensorflow-based Q-Learning model which aims to generate the algorithm able to have the best result in the calc of the distribution of prime numbers.

For simplicity and performance rapidity, the project could begin to evaluate numbers from 2 to 10.000, maybe adding a zero per advancement.

## Collaborators
Ideated by Riccardo Cecchini \<rcecchini.ds@gmail.com>

# How works
The "game" we will create is based on fixed cycles, some costants and fixed variables that the machine can interpret and alter using the operations at its disposition, with the possibility to use additional variables.

The game looks like a vim editor and its input is represented by integers. The virtual screen is represented by a scrollable grid where every "pixel" is represented by an array of 3 items: **isSelected**, **class**, **value**.
- **isSelected**: 0 normally, 1 if the focus is on this element.
- **class**: The type of the value. 0 for labels, 1 for numbers.
- **value**: The ID of the label or the value of the number.

## Possible actions
- **right**: Move the cursor on the right
- **down**: Change the current pixel to the next value

## Variables and costants rappresentation
Are used two pixels for representing a store: the type and the value.
The type: in the labeling point of view is composed by two chars, representing the type of the store, if it's a decimal or a boolean (d or b), and if it's a constant or a variable (# or \$).

The value it's the position of the store in the stack.

- For example a decimal constant at position 2 is written: `d# 2`
- A boolean variable at position 1 is written: `b$ 1`

The total store type are finally 4:
1.   `d#`
2.   `b#`
3.   `d$`
4.   `b$`

## Instruction representation
A line is used to represent an instruction. It's composed by:
`lineNumber storeValueTo operation storeArg1 storeArg2`

An example could be a sum:
`1 d$1 SUM d$0 d#2 `, which is the equivalent of `d$1 = d$0 + d#2`

During the selection of the first store is possible to select just variables or the label `IF`. In this last case the continuation of the instruction change radically: in fact after an IF there is just the store boolean.

Totally, a line is at a maximum long 8 pixels.

## IF and END statements
`IF` and `END` statements are unique instructions different than normal operations lines.
In fact they take just zero or one argument, meaning the boolean store which conditions the behavior of the `IF`. During the writing of the `IF` the next instructions enter in a new context. For exiting from current context is necessary to use the `exit` action, which set the next line to `END` instruction. The `END` instruction could be also used for ending the entire script.

## Loops
For the moment loops are not supported, but they could be necessary for more precision in the best result possible. Anyway, it's a complication which must be handled delicately.

## The cycle
The cycle is composed by these variable (this is pseudo-code)

```
decimal #zero = 0
decimal #one = 1

bool #false = false
bool #true = true

decimal #step = 2 // the cycle starts from 2
decimal #i // #step - 1
decimal #numPrimes = 0 // the current number of found primes
decimal #lastPrime = 0 // the difference between the current step and the last found prime

decimal #primeProb = 0 // the current probability of a prime in the stack
decimal #notPrimeProb = 1 // 1 - #primeProb

decimal #predictedNotPrimeProb = 0 // the predicted probability of a not prime
decimal #predictedPrimeProb // 1 - #predictedNotPrimeProb

// Calculate #predictedNotPrimeProb if it's a prime
decimal #ifPrimePredictNotPrimeProb
decimal #ifPrimePredictPrimeProb

decimal #quanto // 1 / step

bool $isPrime = false

while(#step < upTo){

  #i = step - 1  
  #quanto = 1 / #step

  #ifPrimePredictNotPrimeProb = #predictedNotPrimeProb
  if (#ifPrimePredictNotPrimeProb == 0){
      #ifPrimePredictNotPrimeProb = #quanto
  }
  else {
      decimal primeStamp = #quanto
      primeStamp *= #predictedPrimeProb
      #ifPrimePredictNotPrimeProb += primeStamp
  }    

  decimal #ifPrimePredictPrimeProb = 1 - #ifPrimePredictNotPrimeProb

  ///////////////////////////////////////
  /////////// GAME CONTENT //////////////
  ///////////////////////////////////////

  if($isPrime){
    #numPrimes += 1
    #lastPrime = 0

    #predictedNotPrimeProb = #ifPrimePredictNotPrimeProb
    #predictedPrimeProb = #ifPrimePredictPrimeProb
  }

  #primeProb = #numPrimes / #i
  #notPrimeProb = 1 - #primeProb

  #lastPrime += 1
  #step += 1
}

```

### Additional stores
These are stores which can be added later.

```
decimal #probNextPrime
decimal #probNextNotPrime
decimal #predictNextPrime
decimal #predictNextNotPrime

decimal #nextPrime = 0
decimal #nextNotPrime = 0

cycle {

  #probNextPrime = 1 / #primeProb
  #probNextNotPrime = 1 / (1 - #primeProb)
  #predictNextPrime = 1 / #predictedPrimeProb
  #predictNextNotPrime = 1 / #predictedNotPrimeProb

  //////// STUFF //////////

  if($isPrime){
    #nextPrime += #predictNextPrime
    #nextNotPrime -= 1
  }
  else {
    #nextNotPrime += #predictNextNotPrime
    #nextPrime -= 1
  }
}
```

## Operations

Divided by the type returned

### Neutral
- `ASSIGN`: store's value assigned to another store
- `DEFAULT`: if the assign variable doesn't exists, set the argument value

They both take just 1 argument of the same type.

### Decimal:
- `ADD`: addition
- `SUB`: subtract
- `MUL`: multiply
- `DIV`: divide

### Boolean:
- `NOT`: invert bool value [boolean]
- `CMP`: compare
- `GT`: great than [decimal]
- `GET`: great or equal than [decimal]

#### Removed:
- `LT`: less than [decimal]
- `LET`: less or equal than [decimal]

The reason of `LT` and `LET` removement it's that the same result could be obtained with an `GT` and `GET` inverting the arguments order.


