from User import User
from Database import Database as db
from Data import Data
from _tracemalloc import stop

def accountMenu(current_user):
    choice = -1
    
    while choice != '3':
        print('\n')
        print("1. Enter your Basic Symptoms for Heart-Disease prediction")
        print('2. Exit')
        
        choice = input('Enter your choice: ')
        if(choice == '1'):
            user_id = input('Enter your User-Id (Verification): ')
            try:

               while True:
                   age = float(input('Enter your Age: '))
                   if 1 <= (age) < 125:
                       break
                   print("")
                   print('Enter between 1 to 125 only.\n')


               while True:
                   sex = float(input('Sex (Type - 1 = male; 0 = female): '))
                   if 0 <= (sex) <= 1:
                       break
                   print("")
                   print('Enter only 1 and 0.\n')
               while True:
                   cp = float(input('Chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic): '))
                   if 1 <= (cp) <= 4:
                       break
                   print("")
                   print('Enter includ. 1 to 4 only.\n')


               while True:
                   bp = float(input("Blood_pressure - resting blood pressure (in mm 100-200): "))
                   if 100 <= (bp) <= 200:
                       break
                   print("")
                   print('Enter between 100 to 200 only.\n')

               while True:
                   sc = float(input('Serum cholestoral in mg/dl(200-400): '))
                   if 200 <= (sc) <= 400:
                       break
                   print("")
                   print('Enter between 200 to 400 only.\n')

               while True:
                   fs = float(input('Fasting blood sugar > 120 mg/dl (1 = true; 0 = false): '))
                   if 0 <= (fs) <= 1:
                       break
                   print("")
                   print('Enter only 1 and 0.\n')

               while True:
                   re = float(input('Resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy): '))
                   if 0 <= (re) <= 2:
                       break
                   print("")
                   print('Enter only 0, 1 and 2.\n')

               while True:
                   mh = float(input('Max_heart_rate (150-200): '))
                   if 150 <= (mh) <= 210:
                       break
                   print("")
                   print('Enter between 150 to 210 only.\n')

               while True:
                   ig = float(input('induced_angina - exercise induced angina (1 = yes; 0 = no): '))
                   if 0 <= (ig) <= 1:
                       break
                   print("")
                   print('Enter only 1 and 0.\n')

               while True:
                   st = float(input('ST depression induced by exercise relative to rest (0.0-4.0 in decimals): '))
                   if 0 <= (st) <= 4:
                       break
                   print("")
                   print('Enter between 0.0 to 4.0 only.\n')


               while True:
                   stg = float(input('the slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping): '))
                   if 1 <= (stg) <= 3:
                       break
                   print("")
                   print('Enter 1, 2 and 3 only.\n')

               while True:
                   nv = float(input('no_of_vessels - number of major vessels (0-3) : '))
                   if 0 <= (nv) <= 3:
                       break
                   print("")
                   print('Enter between 0 to 3 only.\n')

               while True:
                   th = float(input('thal - 3 = normal; 6 = fixed defect; 7 = reversable defect: '))
                   if (th) == 3:
                       break
                   elif (th) == 6:
                       break
                   elif (th) == 7:
                       break
                   print("")
                   print('Enter 3, 6 and 7 only.\n')
            except ValueError:
               print("")
               print("Not a numeric value, enter only integer & float values!")
               accountMenu(current_user);

            new_data = Data(user_id, age, sex, cp, bp, sc, fs, re, mh, ig, st, stg, nv, th)

            if ((User.authenuser_id(user_id)) and (new_data.save())):
                print("")
                print('Your Symptoms Got Submitted!, Check Result on Vaticination Portal')
            else:
                print("")
                print('Sorry Symptoms not submitted! Note: Check User ID')

        elif choice == '2':
            exit()
        else:
            print('Invalid option !')

def mainMenu():
    logo = ''' ******************* WELCOME TO VATICINATION - vaticination.ga ****************
               
       *************** Heart Disease Prediction System  ***********
            
          ************* MAKE SURE THAT INTERNET IS WORKING *********** '''

    print(logo)
    choice  = -1    
    current_user = -1

    while choice != '4':
        print('\n')
        print('1.Sign Up')
        print('2.Login Up')
        print('3.Quit')

        choice = input('Enter your choice: ')

        if choice == '1':
            first_name = input('First Name: ').upper()
            last_name = input('Last Name: ').upper()
            user_id = input('User id (Case-Sensitive): ')
            while True:
                password = input('Password (Min. 6-12 Digit): ')
                if 6 <= len(password) < 12:
                    break
                print("")
                print ('The password must be between 6 and 12 characters.\n')            
            address = input('Enter Address: ').upper()
            city = input('City: ').upper()
            state = input('State: ').upper()
            pincode = int(input('Pincode: '))
            phone = int(input('Phone No.: '))
            new_user = User(first_name, last_name,user_id, password, address, city, state, pincode, phone)
            
            if new_user.save():
                print("")
                print('Thanks for signing up!')
                current_user = user_id
                accountMenu(current_user);
            else:
                print('Cannot create an account for you')
                
        elif choice == '2':
            attempts = 1
            while attempts <=3:
                print("")
                print("Attempt %d: "%(attempts))
                user_id = input('User id (Case-Sensitive): ')
                password = input('Password: ')

                if (User.authenticate(user_id,password)):
                    current_user = user_id
                    accountMenu(current_user);
                    break            
                else:
                    print('Invalid credentials!')
                attempts += 1
            else:
                print('Max sign in attempts reached')
        elif choice == '3':
            exit()
        else:
            print('Invalid option!')
mainMenu();