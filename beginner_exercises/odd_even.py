def odd_even():
    """Take number, print out appropriate message to user whether number is even or odd"""
    input_prompt = 'Please enter a number: '
    number = int(input(input_prompt))
    r = number % 2
    if r is 0 and (number%4) is 0:
        print('this number is divisible by 4!')
    elif r is 0:
        print(str(number) + ' is an even number.')
    else:
        print(str(number) + ' is an odd number.')

def num_check():
    """Take two numbers, check if first number is divisible by second number"""
    num = int(input('Please enter the number you want to check: '))
    check = int(input('Please enter the number you want to divide by: '))
    if (num % check) is 0:
        print('Yes, ' + str(num) + ' is divisible by ' + str(check) + '.')
    else:
        print('No, ' + str(num) + ' is not divisible by ' + str(check) + '.')
