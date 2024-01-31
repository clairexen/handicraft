module Counter where
import CLaSH.Prelude

counter state (set, setval) = (state', state)
  where state' = if set then setval else state + 1

topEntity :: Signal (Bool, Unsigned 8) -> Signal (Unsigned 8)
topEntity = mealy counter 0

{-# ANN topEntity
  (defTop
    {
      t_name    = "counter",
      t_inputs  = [ "set", "din" ],
      t_outputs = [ "dout" ]
    })
  #-}
