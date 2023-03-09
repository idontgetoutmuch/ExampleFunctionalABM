{-# LANGUAGE Arrows #-}
{-# LANGUAGE Strict #-}
module Main where

import           Data.IORef
import           System.IO
import           Text.Printf

import           Control.Monad.Random
import           Control.Monad.Reader
import           Control.Monad.Trans.MSF.Random -- from Dunai
import           Data.Array.IArray
import           FRP.BearRiver
import qualified Graphics.Gloss as GLO
import qualified Graphics.Gloss.Interface.IO.Animate as GLOAnim
import           Data.MonadicStreamFunction.InternalCore

import           Graphics.Gloss.Export
import           Data.List (unfoldr)

import qualified Data.ByteString.Lazy as LBS
import qualified Data.Csv             as Csv


data SIRState = Susceptible | Infected | Recovered deriving (Show, Eq)

type Disc2dCoord  = (Int, Int)
type SIREnv       = Array Disc2dCoord SIRState

type SIRMonad g   = Rand g
type SIRAgent g   = SF (SIRMonad g) SIREnv SIRState

type SimSF g = SF (SIRMonad g) () SIREnv

data SimCtx g = SimCtx
  { simSf    :: !(SimSF g)
  , simEnv   :: !SIREnv
  , simRng   :: g
  , simSteps :: !Integer
  , simTime  :: !Time
  }

contactRate :: Double
contactRate = 5.0

infectivity :: Double
infectivity = 0.10

illnessDuration :: Double
illnessDuration = 15.0

agentGridSize :: (Int, Int)
agentGridSize = (51, 51)

winSize :: (Int, Int)
winSize = (800, 800)

cx, cy, wx, wy :: Int
(cx, cy)   = agentGridSize
(wx, wy)   = winSize

cellWidth, cellHeight :: Double
cellWidth  = (fromIntegral wx / fromIntegral cx)
cellHeight = (fromIntegral wy / fromIntegral cy)

winTitle :: String
winTitle = "Agent-Based SIR on 2D Grid"

main :: IO ()
main = do
  hSetBuffering stdout NoBuffering

  let visualise = True
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

transformToWindow :: Disc2dCoord -> (Float, Float)
transformToWindow (x, y) = (x', y')
  where
    rw = cellWidth
    rh = cellHeight

    halfXSize = fromRational (toRational wx / 2.0)
    halfYSize = fromRational (toRational wy / 2.0)

    x' = fromRational (toRational (fromIntegral x * rw)) - halfXSize
    y' = fromRational (toRational (fromIntegral y * rh)) - halfYSize

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

runSimulationUntil :: RandomGen g
                   => Time
                   -> DTime
                   -> SimCtx g
                   -> [(Double, Double, Double)]
runSimulationUntil tMax dt ctx0 = runSimulationAux 0 ctx0 []
  where
    runSimulationAux :: RandomGen g
                      => Time
                      -> SimCtx g
                      -> [(Double, Double, Double)]
                      -> [(Double, Double, Double)]
    runSimulationAux t ctx acc
        | t >= tMax = acc
        | otherwise = runSimulationAux t' ctx' acc'
      where
        env  = simEnv ctx
        aggr = aggregateStates $ elems env

        t'   = t + dt
        ctx' = runStepCtx dt ctx
        acc' = aggr : acc

appendLine :: Csv.ToRecord a => Handle -> a -> IO ()
appendLine hndl line = LBS.hPut hndl (Csv.encode [Csv.toRecord line])

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
    nextFrame :: RandomGen g
              => IORef (SimCtx g)
              -> Float
              -> IO GLO.Picture
    nextFrame ctxRef _ = do
      ctx <- readIORef ctxRef

      let ctx' = runStepCtx dt ctx
      writeIORef ctxRef ctx'

      return $ ctxToPic ctx

mkSimCtx :: RandomGen g
         => SimSF g
         -> SIREnv
         -> g
         -> Integer
         -> Time
         -> SimCtx g
mkSimCtx sf env g steps t = SimCtx {
    simSf    = sf
  , simEnv   = env
  , simRng   = g
  , simSteps = steps
  , simTime  = t
  }

evaluateCtxs :: RandomGen g => Int -> DTime -> SimCtx g -> [SimCtx g]
evaluateCtxs n dt initCtx = unfoldr g (initCtx, n)
  where
    g (c, m) | m < 0 = Nothing
                   | otherwise = Just (c, (runStepCtx dt c, m - 1))

animation :: RandomGen g => [SimCtx g] -> DTime -> Time -> SimCtx g
animation ctxs dt t = ctxs !! (floor (t / dt))

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

initAgentsEnv :: (Int, Int) -> ([(Disc2dCoord, SIRState)], SIREnv)
initAgentsEnv (xd, yd) = (as, e)
  where
    xCenter = floor $ fromIntegral xd * (0.5 :: Double)
    yCenter = floor $ fromIntegral yd * (0.5 :: Double)

    sus = [ ((x, y), Susceptible) | x <- [0..xd-1],
                                    y <- [0..yd-1],
                                    x /= xCenter ||
                                    y /= yCenter ]
    inf = ((xCenter, yCenter), Infected)
    as = inf : sus

    e = array ((0, 0), (xd - 1, yd - 1)) as

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

sirAgent :: RandomGen g => Disc2dCoord -> SIRState -> SIRAgent g
sirAgent coord Susceptible = susceptibleAgent coord
sirAgent _     Infected    = infectedAgent
sirAgent _     Recovered   = recoveredAgent

susceptibleAgent :: RandomGen g => Disc2dCoord -> SIRAgent g
susceptibleAgent coord
    = switch
      -- delay the switching by 1 step, otherwise could
      -- make the transition from Susceptible to Recovered within time-step
      (susceptible >>> iPre (Susceptible, NoEvent))
      (const infectedAgent)
  where
    susceptible :: RandomGen g
                => SF (SIRMonad g) SIREnv (SIRState, Event ())
    susceptible = proc env -> do
      makeContact <- occasionally (1 / contactRate) () -< ()

      if not $ isEvent makeContact
        then returnA -< (Susceptible, NoEvent)
        else (do
          let ns = neighbours env coord agentGridSize moore
          -- let ns = allNeighbours env
          s <- drawRandomElemS -< ns
          case s of
            Infected -> do
              infected <- arrM (const (lift $ randomBoolM infectivity)) -< ()
              if infected
                then returnA -< (Infected, Event ())
                else returnA -< (Susceptible, NoEvent)
            _       -> returnA -< (Susceptible, NoEvent))

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
      recovered <- occasionally illnessDuration () -< ()
      if isEvent recovered
        then returnA -< (Recovered, Event ())
        else returnA -< (Infected, NoEvent)

recoveredAgent :: RandomGen g => SIRAgent g
recoveredAgent = arr (const Recovered)

drawRandomElemS :: MonadRandom m => SF m [a] a
drawRandomElemS = proc as -> do
  r <- getRandomRS ((0, 1) :: (Double, Double)) -< ()
  let len = length as
  let idx = fromIntegral len * r
  let a =  as !! floor idx
  returnA -< a

randomBoolM :: RandomGen g => Double -> Rand g Bool
randomBoolM p = getRandomR (0, 1) >>= (\r -> return $ r <= p)

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

neumann :: [Disc2dCoord]
neumann = [ topDelta, leftDelta, rightDelta, bottomDelta ]

moore :: [Disc2dCoord]
moore = [ topLeftDelta,    topDelta,     topRightDelta,
          leftDelta,                     rightDelta,
          bottomLeftDelta, bottomDelta,  bottomRightDelta ]

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

aggregateStates :: [SIRState] -> (Double, Double, Double)
aggregateStates as = (susceptibleCount, infectedCount, recoveredCount)
  where
    susceptibleCount = fromIntegral $ length $ filter (Susceptible==) as
    infectedCount = fromIntegral $ length $ filter (Infected==) as
    recoveredCount = fromIntegral $ length $ filter (Recovered==) as
