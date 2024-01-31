
reg x 4
reg y
reg z

# write x[]:
#   x[0] <= 42
#   x[1] <= 43
#   x[2] <= 44
#   x[3] <= 45

ld x
st $y

ld 42
st $z

loop:

ld $y
sub x+4
bc end

ld $z
st $$y
add 1
st $z

ld $y
add 1
st $y

b loop

end:
b end

