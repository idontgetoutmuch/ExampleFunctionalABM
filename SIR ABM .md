# Agent based SIR modelling

In 1978, anonymous authors sent a note to the British Medical Journal reporting an influenza outbreak in a boarding school in the north of England ([1]). The chart below shows the solution of the SIR (Susceptible, Infected, Record) model with parameters which give roughly the results observed in the school.

![Screen%20Shot%202023-04-03%20at%206.28.28%20pm.png](attachment:Screen%20Shot%202023-04-03%20at%206.28.28%20pm.png)

[2] give a simple model of the spread of an infectious disease using ordinary differential equations (ODE). Individuals move from being susceptible (S) to infected (I) to recovered (R).

An alternative to modelling via differential equations is to use an agent-based model (ABM). An ABM is a computational model which uses a bottom-up approach to simulate the actions and interactions of the system’s constituent units (agents) to capture the global system behaviour. Modelling via ABMs has become popularised as a result of computational and information processing advancements in technology, mathematics, game theory, and physics that occurred during the twentieth century, including Von Neumann’s cellular automata, Conway’s Game of Life, and Holland’s genetic algorithms [1].

[1] Melissa Tracy, Magdalena Cerda and Katherine Keyes. Agent-Based Modeling in Public Health: Current Applications and Future Directions. Annual Review of Public Health, 39:77–94, January 2018.

## What is ABM?

In agent based modelling, the autonomous agents are assigned characteristics (e.g.age) and governing rules which determine their interaction with other agents and the environment. These interactions can result in emergence of complex behavior patterns and provide valuable information about the dynamics of the real-world system being emulated [2]. 


The main advantages of ABM are:
- Their stochastic nature which may result in unanticipated system dynamics or behaviours unlike those of the deterministic ODE models. 
- Their ability to model experiments that may not be possible to conduct in the real world due to limitations such as cost.
- Their flexibility in the scale of the model [2].



[2] Bonabeau Eric. Agent-based modelling: Methods and techniques for simulating human systems. Proceedings of the National Academy of Sciences, 99: 7280-7287, May 2002.


Traditionally, ABM simulations are implemented as object-oriented programs due to the intuitive mapping of objects to agents. However, Functional Reactive Programming (FRP) can also be used as it can guarantee the reproducibility of the simulation at compile time and avoid specific run-time bugs - which cannot be achieved in traditional object-oriented languages such as C#. 

## What is Functional Reactive Programming?

Functional reactive programming (FRP) provides a method of implementing systems with continuous and discrete time-semantics in pure functional languages. FRP
successfully applied in many domains, such as robotics or user interfacing. 

The central abstraction in FRP is a signal - which is a value that varies overtime. These signals can be composed directly or by composing signal functions. FRP libaries generally use arrows to implement functionality rather than monads due to greater efficiency and modularity. Arrows are a superset of monads, hence have very similar uses but are more restrictive (see [here](https://pdf.sciencedirectassets.com/272990/1-s2.0-S1571066106X02438/1-s2.0-S1571066106001666/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJ7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIFDg9QZeU9kXNykNtK1Y3q77oHuug09p5ti63It1vccTAiBFKp5vMEkDr9v8O6lU%2FGJEBt1WPAktubolt%2FM7Ft0eRyqzBQhXEAUaDDA1OTAwMzU0Njg2NSIMm1WFiIp%2BgTN43MxSKpAF4WtU%2BhIjuCHa7o1OJ%2FJ5MHrwqLRo%2BnhAJ7pK%2BeBHF6xEGRXFSUKaELPIEgO6CAoqLitoqxUvWasKIWiBouge9w%2FZhcoQKARlX6lw6dYnQktYC%2BzakAwobpJa1s2LEWugYNdL9j2bfEZZJ3AtIqbUlwtvbBhqAEyKxkEQHlolc9cHyaq17Vv6%2FgfPDeE%2BqM6cM5I025Q4nQcAK4cQMzb66%2BxiIVpcAGVR8gSprc7GfKIpQlVO17l7aPJ33%2FDrrNMG28QydoXVKseJN2bkGk6%2BnjlP0TItMeFC5r%2FEtx%2BpgYTS%2BpnS%2FPxWeCDUMq4HfTNB9qVMipypeP98oeGMR92%2BNjp2uhF9X2a%2BHJbdOOQVV3IxKsT5JjSJ%2FzJhp6ieJ%2BPM7%2BCSIsCQ7Udf%2FaDo0KeX6KsGtCNFh0zJFwNyel6sx1b5yPXG8t7rgC2KAGV1kaYpggNZKMoGHFDt0HUOQCIan0hiP9vW8GvX0HjI1%2FH3Kajx0qVKnbky%2FbNfGmj9gMCJBfxYpRNvO0as%2FtWBngW7QQHIQR2opY400yJxE%2BdVFCgfCDaYBuDl1sckh7O%2FbDKT%2FQtEHmRlUdYkg8BIHYsqC%2FE3nt5Sqo83JGpaL7%2F0R0TyioO8EazXMwWVDK%2B%2Fbn4AhmoxIEPuAhw%2FgkfAUtn9o7a3iZF%2BHT%2Fv6gGxKLJkI40OzkM3nTlHmd93Yceamm%2Bj%2BHUcMzkqyUxyVUGQLEuTFWJ8DrFGsLqE%2F0h9A4y1sGV48%2BYXiiRipqitUyx7fW%2F1JRuY8sPsJmLc4CnIihYXIhIhzm06xqmouOaOEuSlAFvnAuvy6f09mD8VCC8H2E%2BtqkmUciVlg4fVM3mMRHdePbtNJ8HxyaN6143%2BMD8GVVkwzdqloAY6sgFkA1yE%2BQykIX6iO2s8d%2Bsr4dc1A4L23Kw4TymTEusts%2BVxCXWhF2CU23C3MxIddMHinkuiv8u6J6tcehBM05NTGXotQgTLeL4IASQ6%2BNdj6YWRjHZ5A77E8qBMap%2FWzGzpDXLAJ4hDZWktDQ4KMpW446reCDlw0ViRPJTzaSPsCcuElkjGI34WpARqHLN149%2FSs%2FcIFWBqwhG0%2F93p3zgWt5gkv9u5lbg6ZOYZ1Bjgzm9z&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230309T062929Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY3TG3MPND%2F20230309%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=ae795e67238960dc7bcdc97d21a32e3c73d229d94f6ccf87eecdd2eb4cb17ccb&hash=6109688cb98c98a108b74ea3b2556b59fcada4fd8ea4d0942973369e63ce2e0e&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1571066106001666&tid=spdf-0f923ac0-927a-479a-84e6-8376df53a474&sid=749e13d39ace604f5e2bba26f52d6131b5ccgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=04125300565756535e54&rr=7a514386fa4faafb&cc=au) for more on arrows vs monads).

The central concept of arrowised FRP is the Signal Function (SF) which are functions that have awareness of the passing of time through the timestep of the system. This concept of using time-varying functions as a method of handling agent-based models is essential due to their time-dependent nature (see more [here](https://www.cs.yale.edu/publications/techreports/tr1049.pdf)).


Futher reading on FRP concepts https://ivanperez.io/papers/2016-HaskellSymposium-Perez-Barenz-Nilsson-FRPRefactored-short.pdf


## SIR model using Functional Reactive Programming

The goal is to use a Functional Reactive Programming (FRP) approach in Haskell to simulate a SIR (Susceptible, Infected, Recovered) model - which is a simple compartmental model. The general result within such a SIR model is as below:
![Screen%20Shot%202023-02-15%20at%203.49.56%20pm.png](attachment:Screen%20Shot%202023-02-15%20at%203.49.56%20pm.png)

The following sections describe the approach taken to achieve the above trend using a FRP approach in Haskell.

## Setting up SIR simulation using Haskell

#### Language Extensions
Language extensions are used to enable certain features in Haskell. In this case, we require the following features:
- `Arrows` : to support the arrow notations
- `Strict` : to allows functions parameters to be evaluated before calls
- `FlexibleContexts` : loosens restrictions on what constraints can be used in a typeclass


```haskell
{-# LANGUAGE Arrows #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE FlexibleContexts #-}
```

#### Loading Modules
Various modules required for the different aspect of simulation must be loaded.


```haskell
import           Data.IORef
import           System.IO
import           Text.Printf

import           Control.Monad.Random
import           Control.Monad.Reader
import           Control.Monad.Trans.MSF.Random
import           Data.Array.IArray
import           FRP.BearRiver
import qualified Graphics.Gloss as GLO
import qualified Graphics.Gloss.Interface.IO.Animate as GLOAnim
import           Data.MonadicStreamFunction.InternalCore
import           Data.List (unfoldr)
import           Graphics.Gloss.Export

import qualified Data.ByteString.Lazy as LBS
import qualified Data.Csv             as Csv
import Plots.Axis  (Axis, r2Axis)
import Plots.Axis.Render (renderAxis, r2AxisMain)
import Plots.Types (display)

import Control.Lens ((&~), (.=))
import Diagrams.Backend.Cairo (B, Cairo)
import Diagrams.TwoD.Types (V2)
import Diagrams.Core.Types (QDiagram)
import IHaskell.Display.Diagrams
import IHaskell.Display.Juicypixels hiding (display)
import Plots (scatterPlot, key)

import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Vector as V
```

#### SIR States as a Algebraic Data type
The main governing property of the agents within the SIR simulation is their state - which can be either Susceptible, Infected or Recovered. Here, these possible states are described as a Algebraic data type (ADT) to ensure ease of assignment later.

Additionally, a deriving (Show, Eq) clause is used to ensure the complier automatically generates instances of the Eq and Show classes for the ADT.


```haskell
data SIRState = Susceptible | Infected | Recovered deriving (Show, Eq)
```

#### Define 2D environment
In this model, a N x A grid is used to define the spatial aspect of the model. Here, the discrete 2D environment is simply defined with a tuple.

If this spatial model was described using differential equations, then partial differential equations would be required to model the time dimension and the two space (x and y) dimensions. Such differential equations are amenable to numerical methods, however the additional complexity caused by increasing dimensionality is more easily modelled via ABM. 


```haskell
type Disc2dCoord  = (Int, Int)
```

#### Define types to store agent's environment
In this model, all posiitions of the grid contain an agent and this agent is either susceptible, infected or recovered at all times. This information must be retrievable during simulation, hence the type `SIREnv` is defined to allow agents to store their location coordinate and their state in an array.


```haskell
type SIREnv       = Array Disc2dCoord SIRState
```

Next, to aid readability, we define a type synonym for `Rand` monad which allows us to obtain random numbers parameterised over the random number generator `g`


```haskell
type SIRMonad g   = Rand g 
```

#### Define the SIR Agent
`SIRAgent` is used to define Agents as Signal functions which receives the SIR states of all agents as input and outputs the current SIR state of the agent. It also returns the output in a `SIRMonad` context - which is building the monad over a rand monad.


```haskell
type SIRAgent g   = SF (SIRMonad g) SIREnv SIRState
```

`SimSF` type is similar to `SIRagent` above, however it does not receive any inputs and just outputs the current location and state of the agent. This is useful in instances where states of neighbours must be accessed by an agent to determine whether infection occurs. 



```haskell
type SimSF g = SF (SIRMonad g) () SIREnv
```

#### Defining simulation context
A simulation context data struct which contains the various parameters (e.g. simulation time) is defined in terms of their type. The ! is used to specific it is a strictness declaration and to avoid space leaks.


```haskell
data SimCtx g = SimCtx 
  { simSf    :: !(SimSF g)
  , simEnv   :: !SIREnv
  , simRng   :: g
  , simSteps :: !Integer
  , simTime  :: !Time
  }
```

### Simulation Rules
In this simulation, the rules are:
- There is a population of size N in which all agents are either Susceptible, Infected or Recovered at a particular time.
- Initially, there is at least one infected person in the population.
- People interact with each other on average with a given rate of β per time-unit.
- People become infected with a given probability γ when interacting with an infected person.
- When infected, a person recovers on average after δ time-units.
- An infected person is immune to further infections (no reinfection).
- The 2D environment has either Moore or von Neumann neighbourhood.
- Agents are either static or can move freely around the cells.
- The cells allow either single or multiple occupants.
- Agents can read the states of all their neighbours which tells them if a neighbour is infected or not.

#### Simulation Parameters set up
To enforce the simulation rules, various simulation parameters are defined.


```haskell
-- contact rate or β 
contactRate :: Double
contactRate = 5.0

-- Infection rate or γ
infectivity :: Double
infectivity = 0.025

-- Recovery rate or δ
illnessDuration :: Double
illnessDuration = 15.0

-- 2D grid size
agentGridSize :: (Int, Int)
agentGridSize = (51, 51)
```

The outputs of the simulation will be a CSV file containing the data and an animation, hence parameters relevant to these are set.


```haskell
-- window size for animation 
winSize :: (Int, Int)
winSize = (800, 800)

-- parameters for gif
cx, cy, wx, wy :: Int
(cx, cy)   = agentGridSize
(wx, wy)   = winSize

-- parameters for rendering agents 
cellWidth, cellHeight :: Double
cellWidth  = (fromIntegral wx / fromIntegral cx)
cellHeight = (fromIntegral wy / fromIntegral cy)


-- window title for animation
winTitle :: String
winTitle = "Agent-Based SIR on 2D Grid"
```


<style>/* Styles used for the Hoogle display in the pager */
.hoogle-doc {
display: block;
padding-bottom: 1.3em;
padding-left: 0.4em;
}
.hoogle-code {
display: block;
font-family: monospace;
white-space: pre;
}
.hoogle-text {
display: block;
}
.hoogle-name {
color: green;
font-weight: bold;
}
.hoogle-head {
font-weight: bold;
}
.hoogle-sub {
display: block;
margin-left: 0.4em;
}
.hoogle-package {
font-weight: bold;
font-style: italic;
}
.hoogle-module {
font-weight: bold;
}
.hoogle-class {
font-weight: bold;
}
.get-type {
color: green;
font-weight: bold;
font-family: monospace;
display: block;
white-space: pre-wrap;
}
.show-type {
color: green;
font-weight: bold;
font-family: monospace;
margin-left: 1em;
}
.mono {
font-family: monospace;
display: block;
}
.err-msg {
color: red;
font-style: italic;
font-family: monospace;
white-space: pre;
display: block;
}
#unshowable {
color: red;
font-weight: bold;
}
.err-msg.in.collapse {
padding-top: 0.7em;
}
.highlight-code {
white-space: pre;
font-family: monospace;
}
.suggestion-warning { 
font-weight: bold;
color: rgb(200, 130, 0);
}
.suggestion-error { 
font-weight: bold;
color: red;
}
.suggestion-name {
font-weight: bold;
}
</style><div class="suggestion-name" style="clear:both;">Redundant bracket</div><div class="suggestion-row" style="float: left;"><div class="suggestion-warning">Found:</div><div class="highlight-code" id="haskell">(fromIntegral wx / fromIntegral cx)</div></div><div class="suggestion-row" style="float: left;"><div class="suggestion-warning">Why Not:</div><div class="highlight-code" id="haskell">fromIntegral wx / fromIntegral cx</div></div><div class="suggestion-name" style="clear:both;">Redundant bracket</div><div class="suggestion-row" style="float: left;"><div class="suggestion-warning">Found:</div><div class="highlight-code" id="haskell">(fromIntegral wy / fromIntegral cy)</div></div><div class="suggestion-row" style="float: left;"><div class="suggestion-warning">Why Not:</div><div class="highlight-code" id="haskell">fromIntegral wy / fromIntegral cy</div></div>


#### Defining helper functions for the simulation

To ensure ease of understanding and seperation of functionality, various helper functions are defined for the different aspects of simulation (e.g. for generating the animation).

##### Functions related to the simulation context

`mkSimCtx` and `runStepCtx` are used to initalise the simulation context and update the simulation context respectively.


```haskell
mkSimCtx :: RandomGen g
         => SimSF g
         -> SIREnv
         -> g
         -> Integer
         -> Time
         -> SimCtx g
-- creating the specific SimCtx data struct to return
mkSimCtx sf env g steps t = SimCtx {
    simSf    = sf
  , simEnv   = env
  , simRng   = g
  , simSteps = steps
  , simTime  = t
  }

runStepCtx :: RandomGen g
           => DTime
           -> SimCtx g
           -> SimCtx g
runStepCtx dt ctx = ctx'
  where
    g   = simRng ctx
    sf  = simSf ctx

    sfReader            = unMSF sf ()
    sfRand              = runReaderT sfReader dt
    ((env, simSf'), g') = runRand sfRand g

    steps = simSteps ctx + 1
    t     = simTime ctx + dt
    ctx'  = mkSimCtx simSf' env g' steps t
```

` evaluateCtxs ` takes the initial context and creates a list of new contexts by running the simulation step.


```haskell
evaluateCtxs :: RandomGen g => Int -> DTime -> SimCtx g -> [SimCtx g]
evaluateCtxs n dt initCtx = unfoldr g (initCtx, n)
  where
    g (c, m) | m < 0 = Nothing
                   | otherwise = Just (c, (runStepCtx dt c, m - 1))
```

##### Functions related to Boundary Conditions

Either Neumann or Moore boundary conditions (BC) are used within simulation. The areas surveyed in these is as represented in the following diagram:
![Screen%20Shot%202023-02-22%20at%205.28.13%20pm.png](attachment:Screen%20Shot%202023-02-22%20at%205.28.13%20pm.png)


The following code is related to enforcing these BC:


```haskell
-- Neumann BC
neumann :: [Disc2dCoord]
neumann = [ topDelta, leftDelta, rightDelta, bottomDelta ]

-- Moore BC
moore :: [Disc2dCoord]
moore = [ topLeftDelta,    topDelta,     topRightDelta,
          leftDelta,                     rightDelta,
          bottomLeftDelta, bottomDelta,  bottomRightDelta ]

-- different Delta values for the BC
topLeftDelta :: Disc2dCoord
topLeftDelta      = (-1, -1)
topDelta :: Disc2dCoord
topDelta          = ( 0, -1)
topRightDelta :: Disc2dCoord
topRightDelta     = ( 1, -1)
leftDelta :: Disc2dCoord
leftDelta         = (-1,  0)
rightDelta :: Disc2dCoord
rightDelta        = ( 1,  0)
bottomLeftDelta :: Disc2dCoord
bottomLeftDelta   = (-1,  1)
bottomDelta :: Disc2dCoord
bottomDelta       = ( 0,  1)
bottomRightDelta :: Disc2dCoord
bottomRightDelta  = ( 1,  1)
```

##### Functions related to using RNG elements

`drawRandomElemS` and `randomBoolM` are related to use of RNG elements in simulation. A specific example is `RandomBoolM` being used to draw a random Boolean to determine whether an infection occurs in the susceptibleAgent function.


```haskell
randomBoolM :: RandomGen g => Double -> Rand g Bool
randomBoolM p = getRandomR (0, 1) >>= (\r -> return $ r <= p)

drawRandomElemS :: MonadRandom m => SF m [a] a
drawRandomElemS = proc as -> do
  r <- getRandomRS ((0, 1) :: (Double, Double)) -< ()
  let len = length as
  let idx = (fromIntegral len * r)
  let a =  as !! (floor idx)
  returnA -< a
```

##### Functions related to defining/governing agent behaviour

Aspects related to the behaviour of agents are defined via the various functions below.

`initAgentsEnv` initialises the simulation environment.


```haskell
initAgentsEnv :: (Int, Int) -> ([(Disc2dCoord, SIRState)], SIREnv)
initAgentsEnv (xd, yd) = (as, e)
  where
    xCenter = floor $ fromIntegral xd * (0.5 :: Double)
    yCenter = floor $ fromIntegral yd * (0.5 :: Double)
    -- populating the grid with susceptible agents everywhere except the centre
    sus = [ ((x, y), Susceptible) | x <- [0..xd-1], 
                                    y <- [0..yd-1],
                                    x /= xCenter ||
                                    y /= yCenter ] 
    -- populating the infected agent at the center                             
    inf = ((xCenter, yCenter), Infected)
    -- list of infected and susceptible agent locations
    as = inf : sus
    -- array of min and max grid size and the list of agent locations 
    e = array ((0, 0), (xd - 1, yd - 1)) as
```

In order to observe the general SIR trend, infection via observing the neighbours must be viable. The neighbours function takes the coordinates and returns the surround SIRstates. `allNeighbours` returns the state of all neighbours or agents.


```haskell
neighbours :: SIREnv 
           -> Disc2dCoord 
           -> Disc2dCoord
           -> [Disc2dCoord] 
           -> [SIRState]
neighbours e (x, y) (dx, dy) n = map (e !) nCoords'
  where
    nCoords  = map (\(x', y') -> (x + x', y + y')) n
    nCoords' = filter (\(nx, ny) -> nx >= 0 && 
                                    ny >= 0 && 
                                    nx <= (dx - 1) &&
                                  ny <= (dy - 1)) nCoords
                                  
 
allNeighbours :: SIREnv -> [SIRState]
allNeighbours = elems
```


<style>/* Styles used for the Hoogle display in the pager */
.hoogle-doc {
display: block;
padding-bottom: 1.3em;
padding-left: 0.4em;
}
.hoogle-code {
display: block;
font-family: monospace;
white-space: pre;
}
.hoogle-text {
display: block;
}
.hoogle-name {
color: green;
font-weight: bold;
}
.hoogle-head {
font-weight: bold;
}
.hoogle-sub {
display: block;
margin-left: 0.4em;
}
.hoogle-package {
font-weight: bold;
font-style: italic;
}
.hoogle-module {
font-weight: bold;
}
.hoogle-class {
font-weight: bold;
}
.get-type {
color: green;
font-weight: bold;
font-family: monospace;
display: block;
white-space: pre-wrap;
}
.show-type {
color: green;
font-weight: bold;
font-family: monospace;
margin-left: 1em;
}
.mono {
font-family: monospace;
display: block;
}
.err-msg {
color: red;
font-style: italic;
font-family: monospace;
white-space: pre;
display: block;
}
#unshowable {
color: red;
font-weight: bold;
}
.err-msg.in.collapse {
padding-top: 0.7em;
}
.highlight-code {
white-space: pre;
font-family: monospace;
}
.suggestion-warning { 
font-weight: bold;
color: rgb(200, 130, 0);
}
.suggestion-error { 
font-weight: bold;
color: red;
}
.suggestion-name {
font-weight: bold;
}
</style><div class="suggestion-name" style="clear:both;">Use bimap</div><div class="suggestion-row" style="float: left;"><div class="suggestion-warning">Found:</div><div class="highlight-code" id="haskell">\ (x', y') -> (x + x', y + y')</div></div><div class="suggestion-row" style="float: left;"><div class="suggestion-warning">Why Not:</div><div class="highlight-code" id="haskell">Data.Bifunctor.bimap ((+) x) ((+) y)</div></div>


Unlike the other states, the recovered state does not generate any event and rather just acts a sink which constantly returns Recovered.


```haskell
recoveredAgent :: RandomGen g => SIRAgent g
recoveredAgent = arr (const Recovered) 
```

# Arrowised FRP

An Arrow is a generalisation of a monad which represents a process that takes as input a value of type b and outputs a value of type c.

![Screen%20Shot%202023-04-06%20at%208.48.22%20pm.png](attachment:Screen%20Shot%202023-04-06%20at%208.48.22%20pm.png)

Arrows are used in FRP as they represent processes - which naturally maps to the definition of signal functions (a process overtime) in FRP. Arrowised programming can be achieved via many different arrows combinators, but in this case to allow for programming in a point-free style and greater code readability, the `proc` keyword along with Paterson's `do` notation is used. 

As an example of arrowised programming via this notation, consider the following code which implements a simple SF that calculates the acceleration of a falling mass given its initial position and velocity:


```haskell
fallingMass :: Double -> Double -> SF () Double
fallingMass p0 v0 = proc _ -> do
    v <- arr (+v0) <<< integral -< (-9.8)
    p <- arr (+p0) <<< integral -< v
returnA -< p
```

The `proc` keyword introduces the arrow notation, and it binds the arrow input to a pattern - the `do` in this case. The integral function - which is a SF - intakes variables and integrates them overtime to determine the current velocity and position. The <<< is a arrow computation combinator,  -< is used to pass an input to an arrow, and <- to bind the result of an arrow computation to a variable. Finally to return a value from an arrow, returnA is used. Overall, the combination of these allows for a more readable arrowised FRP approach. 

To further illustrate the difference in code readability, consider the following example with and without the proc notation:



```haskell
-- computation with proc
proc x -> do
        y <- f -< x+1
        g -< 2*y
        let z = x+y
        t <- h -< x*z
        returnA -< t+z
        
-- equivalent computation but without proc
arr (\ x -> (x+1, x)) >>>
        first f >>>
        arr (\ (y, x) -> (2*y, (x, y))) >>>
        first g >>>
        arr (\ (_, (x, y)) -> let z = x+y in (x*z, z)) >>>
        first h >>>
        arr (\ (t, z) -> t+z)
```

Futher resources for Arrows:

- https://en.wikibooks.org/wiki/Haskell/Understanding_arrows
- https://wiki.haskell.org/Arrow
- https://www.cs.yale.edu/homes/hudak-paul/CS429F04/AFPLectureNotes.pdf

The above notation is used in the following infectedAgent and susceptibleAgent functions.

The function below describes the behaviour of an infected agent. This behaviour is governed by either recovering on average after delta time units or staying infected within a timestep.


```haskell
infectedAgent :: RandomGen g => SIRAgent g
infectedAgent
    = switch
      -- delay the switching by 1 step, otherwise could
      -- make the transition from Susceptible to Recovered within time-step
      (infected >>> iPre (Infected, NoEvent))
      (const recoveredAgent)
  where
    infected :: RandomGen g => SF (SIRMonad g) SIREnv (SIRState, Event ())
    infected = proc _ -> do
      -- illness duration = infected agent
      recovered <- occasionally illnessDuration () -< ()
      if isEvent recovered
        -- otherwise recovered agent as the updated state
        then returnA -< (Recovered, Event ())
        else returnA -< (Infected, NoEvent)
```

`susceptibleAgent` describes the behaviour of a susceptible agent - which is governed by querying the surrounding neighbours and either getting infected based on the parameter γ (event generated) or staying susceptible (no event)


```haskell
susceptibleAgent :: RandomGen g => Disc2dCoord -> SIRAgent g
susceptibleAgent _coord
    = switch 
      -- delay the switching by 1 step, otherwise could
      -- make the transition from Susceptible to Recovered within time-step
      (susceptible >>> iPre (Susceptible, NoEvent))
      (const infectedAgent)
  where
    susceptible :: RandomGen g 
                => SF (SIRMonad g) SIREnv (SIRState, Event ())
    susceptible = proc env -> do
      -- use occasionally to make contact on average 
      makeContact <- occasionally (1 / contactRate) () -< ()

      if not $ isEvent makeContact 
        then returnA -< (Susceptible, NoEvent)
        else (do
          -- take env, the dimensions of grid and neighbourhood info
          --let ns = neighbours env coord agentGridSize moore
          -- queries the environemtn for its neighbours - in this case appears to be all neighbours 
          let ns = allNeighbours env
          s <- drawRandomElemS -< ns -- randomly selects one
          case s of
            Infected -> do
              infected <- arrM (const (lift $ randomBoolM infectivity)) -< ()
              -- upon infection,
              if infected 
                -- event returned which returns in switching into the infected agent SF (to behave as such)
                then returnA -< (Infected, Event ())
                else returnA -< (Susceptible, NoEvent)
            _       -> returnA -< (Susceptible, NoEvent))
```

`sirAgent` defines the behaviour of the agent depending on the initial state. Only the suspectible agent receives the coordinates as the infected and recovered agents do not require this information.


```haskell
sirAgent :: RandomGen g => Disc2dCoord -> SIRState -> SIRAgent g
sirAgent coord Susceptible = susceptibleAgent coord
sirAgent _     Infected    = infectedAgent
sirAgent _     Recovered   = recoveredAgent -- recovered agent ignores gen bc they stay immune
```

The simulationStep function is a closed feedback loop which takes the current signal functions and returns the new agent states. 


```haskell
simulationStep :: RandomGen g
               => [(SIRAgent g, Disc2dCoord)]
               -> SIREnv
               -> SF (SIRMonad g) () SIREnv
simulationStep sfsCoords env = MSF $ \_ -> do
    let (sfs, coords) = unzip sfsCoords 

    -- run all agents sequentially but keep the environment
    -- read-only: it is shared as input with all agents
    -- and thus cannot be changed by the agents themselves
    -- run agents sequentially but with shared, read-only environment
    ret <- mapM (`unMSF` env) sfs
    -- construct new environment from all agent outputs for next step
    let (as, sfs') = unzip ret
        env' = foldr (\(coord, a) envAcc -> updateCell coord a envAcc) env (zip coords as)
        
        sfsCoords' = zip sfs' coords
        cont       = simulationStep sfsCoords' env'
    return (env', cont)
  where
    updateCell :: Disc2dCoord -> SIRState -> SIREnv -> SIREnv
    updateCell c s e = e // [(c, s)]
```

##### Functions related to collating simulation data

`aggregateStates` is used to collate the number of susceptible, infected and recovered agents within the simulation. This function is used both for the animation and the plot.


```haskell
aggregateStates :: [SIRState] -> (Double, Double, Double)
aggregateStates as = (susceptibleCount, infectedCount, recoveredCount)
  where
    susceptibleCount = fromIntegral $ length $ filter (Susceptible==) as
    infectedCount = fromIntegral $ length $ filter (Infected==) as
    recoveredCount = fromIntegral $ length $ filter (Recovered==) as
```

##### Functions for generating  CSV file

`appendLine` is a helper function used to write the counts of the S, I, R states into the CSV file.


```haskell
appendLine :: Csv.ToRecord a => Handle -> a -> IO ()
appendLine hndl line = LBS.hPut hndl (Csv.encode [Csv.toRecord line])
```

`writeSimulationUntil` uses the above auxilliary functions to generate the overall CSV file


```haskell
writeSimulationUntil :: RandomGen g
                     => Time
                     -> DTime
                     -> SimCtx g
                     -> String
                     -> IO ()
writeSimulationUntil tMax dt ctx0 fileName = do
    fileHdl <- openFile fileName WriteMode
    appendLine fileHdl ("Susceptible", "Infected", "Recovered")
    writeSimulationUntilAux 0 ctx0 fileHdl
    hClose fileHdl
  where
    writeSimulationUntilAux :: RandomGen g
                            => Time
                            -> SimCtx g
                            -> Handle
                            -> IO ()
    writeSimulationUntilAux t ctx fileHdl
        | t >= tMax = return ()
        | otherwise = do
          let env  = simEnv ctx
              aggr = aggregateStates $ elems env

              t'   = t + dt
              ctx' = runStepCtx dt ctx

          appendLine fileHdl aggr

          writeSimulationUntilAux t' ctx' fileHdl
```

##### Functions for generating the animation

`visualiseSimulation` generates and updates the animation.


```haskell
visualiseSimulation :: RandomGen g
                    => DTime
                    -> SimCtx g
                    -> IO ()
visualiseSimulation dt ctx0 = do
    ctxRef <- newIORef ctx0

    GLOAnim.animateIO
    
      (GLO.InWindow winTitle winSize (0, 0))
      GLO.white
      (nextFrame ctxRef)
      (const $ return ())

  where
    (cx, cy)   = agentGridSize
    (wx, wy)   = winSize
    cellWidth  = (fromIntegral wx / fromIntegral cx) :: Double
    cellHeight = (fromIntegral wy / fromIntegral cy) :: Double

    nextFrame :: RandomGen g
              => IORef (SimCtx g)
              -> Float 
              -> IO GLO.Picture
    nextFrame ctxRef _ = do
      ctx <- readIORef ctxRef
      
      let ctx' = runStepCtx dt ctx
      writeIORef ctxRef ctx'

      return $ ctxToPic ctx

ctxToPic :: RandomGen g
             => SimCtx g 
             -> GLO.Picture
ctxToPic ctx = GLO.Pictures $ aps ++ [timeStepTxt]
      where
          env = simEnv ctx
          as  = assocs env
          aps = map renderAgent as
          t   = simTime ctx

          (tcx, tcy)  = transformToWindow (-7, 10)
          timeTxt     = printf "%0.1f" t
          timeStepTxt = GLO.color GLO.black $ GLO.translate tcx tcy $ GLO.scale 0.5 0.5 $ GLO.Text timeTxt

renderAgent :: (Disc2dCoord, SIRState) -> GLO.Picture
renderAgent (coord, Susceptible)
    = GLO.color (GLO.makeColor 0.0 0.0 0.7 1.0) $ GLO.translate x y $ GLO.Circle (realToFrac cellWidth / 2)
  where
    (x, y) = transformToWindow coord
renderAgent (coord, Infected)
    = GLO.color (GLO.makeColor 0.7 0.0 0.0 1.0) $ GLO.translate x y $ GLO.ThickCircle 0 (realToFrac cellWidth)
  where
    (x, y) = transformToWindow coord
renderAgent (coord, Recovered)
    = GLO.color (GLO.makeColor 0.0 0.70 0.0 1.0) $ GLO.translate x y $ GLO.ThickCircle 0 (realToFrac cellWidth)
  where
    (x, y) = transformToWindow coord

transformToWindow :: Disc2dCoord -> (Float, Float)
transformToWindow (x, y) = (x', y')
      where
        rw = cellWidth
        rh = cellHeight

        halfXSize = fromRational (toRational wx / 2.0)
        halfYSize = fromRational (toRational wy / 2.0)

        x' = fromRational (toRational (fromIntegral x * rw)) - halfXSize
        y' = fromRational (toRational (fromIntegral y * rh)) - halfYSize
```

`animation` is used to map a list of contexts to time - which is required for the function that produces the gif.


```haskell
animation :: RandomGen g => [SimCtx g] -> DTime -> Time -> SimCtx g
-- bc of the gif time to pictures conversion. Lists of context -> time to context 
animation ctxs dt t = ctxs !! floor (t / dt)
```

##### Functions for running simulation

`runSimulationUntil` runs the overall simulation for the desired duration.


```haskell
runSimulationUntil :: RandomGen g
                   => Time
                   -> DTime
                   -> SimCtx g
                   -> [(Double, Double, Double)]
-- With the max time, time step and initial context, run simulation via the Aux function
runSimulationUntil tMax dt ctx0 = runSimulationAux 0 ctx0 []
  where
    runSimulationAux :: RandomGen g
                      => Time
                      -> SimCtx g
                      -> [(Double, Double, Double)]
                      -> [(Double, Double, Double)]
    runSimulationAux t ctx acc 
        | t >= tMax = acc -- if time step is greater than tmax, 
        | otherwise = runSimulationAux t' ctx' acc'
      where
        env  = simEnv ctx -- 
        aggr = aggregateStates $ elems env

        t'   = t + dt -- increase time by timestep
        ctx' = runStepCtx dt ctx -- get new step context
        acc' = aggr : acc
```

## Main Function

The `main` function below sets up the general simulation via various steps and enables a method of getting output either through an animation or through a CSV file output and a GIF of the animation.


```haskell
main :: IO ()
main = do
  hSetBuffering stdout NoBuffering

  let visualise = False
      t         = 100
      dt        = 0.1
      seed      = 123 -- 42 leads to recovery without any infection

      g         = mkStdGen seed
      (as, env) = initAgentsEnv agentGridSize
      sfs       = map (\(coord, a) -> (sirAgent coord a, coord)) as
      sf        = simulationStep sfs env
      ctx       = mkSimCtx sf env g 0 0

  if visualise
    then visualiseSimulation dt ctx
    else do
      let ts = [0.0, dt .. t]
      let ctxs = evaluateCtxs (length ts) dt ctx
      exportPicturesToGif 10 LoopingForever (800, 800) GLO.white "SIR.gif" ((ctxToPic . (animation ctxs dt)) . uncurry encodeFloat . decodeFloat) (map (uncurry encodeFloat . decodeFloat) ts)
      writeSimulationUntil t dt ctx "SIR_DUNAI_dt001.csv"
```


<style>/* Styles used for the Hoogle display in the pager */
.hoogle-doc {
display: block;
padding-bottom: 1.3em;
padding-left: 0.4em;
}
.hoogle-code {
display: block;
font-family: monospace;
white-space: pre;
}
.hoogle-text {
display: block;
}
.hoogle-name {
color: green;
font-weight: bold;
}
.hoogle-head {
font-weight: bold;
}
.hoogle-sub {
display: block;
margin-left: 0.4em;
}
.hoogle-package {
font-weight: bold;
font-style: italic;
}
.hoogle-module {
font-weight: bold;
}
.hoogle-class {
font-weight: bold;
}
.get-type {
color: green;
font-weight: bold;
font-family: monospace;
display: block;
white-space: pre-wrap;
}
.show-type {
color: green;
font-weight: bold;
font-family: monospace;
margin-left: 1em;
}
.mono {
font-family: monospace;
display: block;
}
.err-msg {
color: red;
font-style: italic;
font-family: monospace;
white-space: pre;
display: block;
}
#unshowable {
color: red;
font-weight: bold;
}
.err-msg.in.collapse {
padding-top: 0.7em;
}
.highlight-code {
white-space: pre;
font-family: monospace;
}
.suggestion-warning { 
font-weight: bold;
color: rgb(200, 130, 0);
}
.suggestion-error { 
font-weight: bold;
color: red;
}
.suggestion-name {
font-weight: bold;
}
</style><div class="suggestion-name" style="clear:both;">Redundant bracket</div><div class="suggestion-row" style="float: left;"><div class="suggestion-warning">Found:</div><div class="highlight-code" id="haskell">ctxToPic . (animation ctxs dt)</div></div><div class="suggestion-row" style="float: left;"><div class="suggestion-warning">Why Not:</div><div class="highlight-code" id="haskell">ctxToPic . animation ctxs dt</div></div>


# Results 

Chart to be finalised. Further details on the results to be added after this.




```haskell
import Plots.Axis
import Control.Lens ((&~), (.=))
import Plots.Axis.Scale
import Plots.Axis.Title
import Plots.Axis.Labels
import Plots.Types
import Diagrams.TwoD.Text
import Diagrams.Core.Measure
```

`getResults`, `decodeCSV` and `addToList` are helper functions used to initally decode the SIR values stored in the CSV file created above, and convert them into List format such that they can be used for plotting


```haskell
decodeCSV :: BL.ByteString -> Either String (V.Vector (Int, Int, Int))
decodeCSV  = decode NoHeader  

getResults :: IO ([(Int, Int)], [(Int, Int)], [(Int, Int)])
getResults = do
    csvData <- BL.readFile "myFile.csv"
    let x = decodeCSV csvData
    case decode NoHeader csvData of
        Left err -> error err
        Right y -> do let (a,b,c) = addToList y
                      let d = zip [1..length a] a --data that I want to use for plotting (trace 1 - S)
                      let e = zip [1..length b] b --data for plotting (line 2)
                      let f = zip [1..length c] c --data for plotting (line 3)
                      pure (d, e, f)

addToList :: V.Vector (Int, Int, Int) -> ([Int], [Int], [Int])
addToList v = unzip3 (V.toList v)
```

`scatterAxis2` is related to using the Plots functionality to create a scatter plot which depicts the resulting SIR Trajectory


```haskell
(d,e,f) <- getResults

scatterAxis2 :: Axis B V2 Double
scatterAxis2 = r2Axis &~ do
    scatterPlot (map (\(x,y) -> (fromIntegral x, fromIntegral y)) d) $ key "S"
    scatterPlot (map (\(x,y) -> (fromIntegral x, fromIntegral y)) e) $ key "I"
    scatterPlot (map (\(x,y) -> (fromIntegral x, fromIntegral y)) f) $ key "R"
    Plots.Axis.xLabel Control.Lens..= "time step"
    Plots.Axis.yLabel Control.Lens..= "No. people infected"
    scaleMode Control.Lens..= Stretch
    renderSize Control.Lens..= Just 500
    titleText Control.Lens..= "Resulting SIR Trajectory"
    tickLabelGap Control.Lens..= 10
    --tickLabelStyle Control.Lens..= fontSize (output 11)


    
--
scatterExample2 = renderAxis scatterAxis2
diagram scatterExample2
```


    
![svg](output_101_0.svg)
    


The GIF below shows the spread of infection overtime

![SegmentLocal](SIR.gif "segment")


```haskell

```
