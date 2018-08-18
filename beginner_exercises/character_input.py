def name_age():
    name_prompt = 'What is your name?'
    age_prompt = 'How old are you?'
    this_year = 2018
    print(name_prompt)
    name = input("")
    print(age_prompt)
    age = int(input(""))
    one_hundred = str((100 - age) + this_year)
    print('How many times do you want to hear this?')
    repeat = int(input(""))
    for numbers in range(repeat):
        print('Hello ' + name + '. You will be 100 years old in the year ' + one_hundred + '.\n')
