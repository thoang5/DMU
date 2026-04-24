import DMUStudent.HW1

#------------- 
# Problem 4
#-------------

# Here is a functional but incorrect answer for the programming question
function f(a, bs)
    final = a * bs[1] # keeps 2 element vector for first column [5, 11]

    for b in bs[2:end] # grabs the rest of the columns
        temp = a * b # multiply the rest of the columns [11, 25]
        for i in eachindex(final) # loops single position in final
            final[i] = max(final[i], temp[i]) # compares the rows to return the elementwise max
        end
    end

    return final
end

# You can can test it yourself with inputs like this
a = [1.0 2.0; 3.0 4.0]
@show a
bs = [[1.0, 2.0], [3.0, 4.0]]
@show bs
@show f(a, bs)

# This is how you create the json file to submit
HW1.evaluate(f, "thomas.hoang@colorado.edu")
