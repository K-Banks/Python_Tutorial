def odd_even():
    """Take number, print out appropriate message to user whether number is even or odd"""
    input_prompt = 'Please enter a number: '
    number = int(input(input_prompt))
    r = number % 2
    if r is 0:
        print(str(number) + ' is an even number.')
    else:
        print(str(number) + ' is an odd number.')
