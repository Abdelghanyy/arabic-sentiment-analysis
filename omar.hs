import Data.List
square x = x*x
cube x = (square x) * x
fourthpower x = square (square x)

istriple x y z = if (square z) == square (x) + square(y)
                 then True
                 else False 


threedifferent x y z = x /=y && y/=z && x/=z

smaller x y | x<= y =x
            | otherwise=y



minThree x y z | x <=y && x <=y =x
               | y<= x && y <= z = y
               | z<= x && z <= y = z 

miinThreee x y z = smaller x (smaller y z)

sumsq 0 = 0
sumsq n = sq + sumsq(n-1) where sq= n^2


fib 0 =0 
fib 1= 1
fib n = (fib n-1) + (fib n-2) 

isprime 2 = True
isprime 3 =True 
isprime x = hasnofactor x (div x 2)

hasnofactor x 2 = mod x 2 /= 0

hasnofactor x n | mod x n ==0 = False
                | otherwise =hasnofactor x (n-2)

last1 :: [a]->a
last1 [] = error  "last1[]"
last1 [x]= x
last1 (x:xs)= last1 xs

occursIn x [] = False
occursIn x (y:ys) | x==y = True
                  | otherwise = occursIn x ys

occurs x []=[]
occurs x (y:ys) | occursIn x y = y:occurs x ys
                | otherwise = occurs x ys

reverse1 [] =[]
reverse1 (x:xs) = (reverse1 xs)++ [x]

maxList [] =0
maxList (x:xs) | maxList xs > x = maxList xs
               | otherwise = x


updatePrices _ [] = []
updatePrices num ((prod,price):xs) =(prod,price+(num/100)* price) : updatePrices num xs


palindrome x= x ==reverse x

prefix [] y = True
prefix (x:xs) (y:ys)| x==y = prefix xs ys
                    | otherwise = False
insert x []= [x]
insert x (y:ys) | x>=y = y:insert x ys
                | otherwise = x:y:ys

sort []=[]
sort (x:xs) = insert x (sort xs)



merge x [] = x
merge [] x = x
merge (x:xs) (y:ys) | x <= y = x:merge xs (y:ys)
                    | otherwise = y:merge (x:xs) ys

sortGT (a1, b1) (a2, b2)
  | a1 < a2 = GT
  | a1 > a2 = LT
  | a1 == a2 = compare b1 b2



