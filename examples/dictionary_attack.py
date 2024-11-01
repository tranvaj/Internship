#load csv  file with ; as separator
tuples =  [tuple(line.strip().split(',')) for line in open('password_database.csv')]
tuples = [tuple for tuple in tuples if tuple[0] == "A21B0299P"]
#print(tuples)

passes = [line for line in open("10-million-password-list-top-10000.txt")]
#print(passes)

#sha256 load
from hashlib import sha256
import string
import itertools
import time

hashes = [(passwd, sha256(passwd.encode("utf-8")).hexdigest()) for passwd in passes]
cracked = []
time_sum = 0
for t in tuples:
    found_passwd = ""
    count = 0
    print("\nCracking password for hash: ", t[1])
    print("Trying dictionary attack")
    time_start = time.time()
    for h in hashes:
        count += 1
        if h[1] == t[1]:
            found_passwd = h[0]
            print("Password is: ", h[0].strip())
            break

    if found_passwd != "":
        print("Time taken: ", time.time() - time_start, " seconds")
        cracked.append((t[1], found_passwd.strip(), count, time.time() - time_start))
        time_sum += time.time() - time_start
        continue

    count = 0
    print("Trying brute force...")
    characters_to_try = [string.digits, string.ascii_lowercase + string.digits]
    for characters in characters_to_try:
        print("Trying with characters: ", characters)
        for length in range(1, 6 + 1):
            for combo in itertools.product(characters, repeat=length):
                # Join the tuple to form a string
                passwd = ''.join(combo) + "\n"
                sha256_passwd = sha256(passwd.encode("utf-8")).hexdigest()
                count = count + 1
                if sha256_passwd == t[1]:
                    found_passwd = passwd
                    print("Password is: ", passwd.strip())
                    print("Time taken: ", time.time() - time_start, " seconds")
                    cracked.append((t[1], found_passwd.strip(), count, time.time() - time_start))
                    time_sum += time.time() - time_start
                    break
                time_spent = time.time() - time_start
                if time_sum + time_spent > 60*12:
                    print("Program taking more than 12 minutes, breaking")
                    break

            if found_passwd != "":
                break
        if found_passwd != "":
            break

    if found_passwd == "":
        print("Password not found")
        cracked.append((t[1], "not cracked", count, time.time() - time_start))



output_filename = "cracked_results_A21B0299P.csv"
with open(output_filename, "w") as f:
    for c in cracked:
        f.write(f"{c[0]};{c[1]};{c[2]};{c[3]}\n")


        




