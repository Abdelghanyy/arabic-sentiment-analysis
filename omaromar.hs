import Data.List
data Ex = Ex Float Float String String deriving (Show,Ord,Eq)
data NewSt = NewSt Float Float String deriving (Show,Ord,Eq)
data Dist = Dist Float NewSt Ex deriving (Show,Ord,Eq) 

euclidean :: NewSt -> Ex -> Dist

euclidean (NewSt stmid stqui firstName) (Ex exmid exqui secondName grade) = Dist (sqrt(((exmid - stmid)^2 )+((exqui - stqui)^2 ))) (NewSt stmid stqui firstName) (Ex exmid exqui secondName grade)

manhattan :: NewSt -> Ex -> Dist

manhattan (NewSt stmid stqui firstName) (Ex exmid exqui secondName grade) = Dist (abs ((stmid-exmid)+(stqui-exqui))) (NewSt stmid stqui firstName) (Ex exmid exqui secondName grade)

dist :: (a -> b -> c) -> a ->b -> c

dist a b c = a b c

all_dists :: (a -> b -> c) -> a -> [b] -> [c]

all_dists f a = map (dist f a)

takeN :: (Eq a, Num a) => a -> [b] -> [b]

takeN 0 (x:xs) = []
takeN a [] = []
takeN a (x:xs) = x:takeN (a-1) xs

sortDist :: Dist -> Dist -> Ordering
sortDist (Dist a b c) (Dist x y z) =if a > x then LT else GT

farthest:: (Eq a, Num a) => (t -> t1 -> Dist) -> a -> [t1] -> t -> [Dist]   
farthest f n exs st =sortBy sortDist (takeN n (all_dists f st exs)) 

closest:: (Eq a, Num a) => (t -> t1 -> Dist) -> a -> [t1] -> t -> [Dist]   

closest f n exs st =sortBy sortDist (takeN n (all_dists f st exs))                             

passgrouping :: [Dist] -> [Dist]
passgrouping [] =[]
passgrouping ((Dist a b (Ex x w y z)):xs)= if z == "pass" then (Dist a b (Ex x w y z)) : passgrouping (xs) else passgrouping (xs)

failgrouping :: [Dist] -> [Dist]
failgrouping [] =[]
failgrouping ((Dist a b (Ex x w y z)):xs)= if z == "fail" then (Dist a b (Ex x w y z)) : failgrouping (xs) else failgrouping (xs)

grouping :: [Dist] -> [[Dist]]
grouping ls = [passgrouping ls , failgrouping ls]

grouped_dists :: (Eq a,Num a)=> (b -> c -> Dist) -> a -> [c] -> b -> [[Dist]]
grouped_dists f n exs st = grouping (closest f n exs st)

maxlength::Foldable t => [t a] -> t a
maxlength [one,two]= if length one <= length two then two else one

mode :: (Eq a,Num a) => (b -> c -> Dist) -> a -> [c] -> b -> [Dist]
mode f n exs st = maxlength (grouped_dists f n exs st)

label_of::[Dist] -> ([Char], [Char])
label_of  ((Dist a (NewSt _ _ name) (Ex x w y z)):xs) =(name,z)

classify :: (Eq a,Num a)=> (b -> c -> Dist) -> a -> [c] -> b -> ([Char],[Char])
classify f n exs st = label_of (mode f n exs st)

classify_all :: (a -> b -> Dist) -> Int -> [b] -> [a] -> [([Char],[Char])]
classify_all f n exs  sts = map (classify f n exs ) sts