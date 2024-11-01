import random
import os
from math import gcd

def bytes2binstr(b, n=None):
    s = ''.join(f'{x:08b}' for x in b)
    return s if n is None else s[:n + n // 8 + (0 if n % 8 else -1)]


RANDOM_INC = 1000

class KnapsackAlg:
    def __init__(self) -> None:
        self.key_len = 250
        self.w_max = 400
        self.w_min = 200
        
    def encrypt(self, input_bytes, pub_key):
        bin_bytes = bytes2binstr(input_bytes)
        padded_bits = self.get_padding_bits_needed(bin_bytes)

        padded_bin_num = self.pad_bits(bin_bytes)
        blocks = [padded_bin_num[x:x+self.key_len] for x in range(0, len(padded_bin_num), self.key_len)]
        cipher_blocks = []
        for block in blocks:
            cipher = 0
            for i in range(len(block)):
                cipher += pub_key[i] * int(block[i])
            cipher_blocks.append(cipher)
        return cipher_blocks, padded_bits
    
    def decipher(self, cipher_blocks, priv_key, p, q, padded_bits):
        p_inverse = pow(p, -1, q)
        messages = []
        for cipher in cipher_blocks:
            deciphered = (cipher * p_inverse) % q
            msg = ['0']*self.key_len
            for i, x in enumerate(reversed(priv_key)):
                if deciphered > 0 and x <= deciphered:
                    deciphered = deciphered - x
                    msg[-i-1] = '1'
            messages.append(''.join(msg))
        padding_removed = ''.join(messages)[:-padded_bits]
        deciphered_message = self.bitstring_to_bytes(''.join(padding_removed))
        return deciphered_message

    def decode_binary_string(self, s):
        return ''.join(chr(int(s[i*8:i*8+8],2)) for i in range(len(s)//8))
    
    def bitstring_to_bytes(self, s):
        return int(s, 2).to_bytes((len(s) + 7) // 8)

    def pad_bits(self, bit_str):
        if len(bit_str) % self.key_len == 0:
            return bit_str, 0
        else:
            padding_bits_needed = self.get_padding_bits_needed(bit_str)
            padded_bit_str = bit_str + '0' * padding_bits_needed

            return padded_bit_str
    def get_padding_bits_needed(self, bit_str):
        return self.key_len - len(bit_str) % self.key_len
    
    def read_params(self, path="."):
        with open(os.path.join(path, "p.txt")) as f:
            self.p = int(f.read())
        with open(os.path.join(path, "q.txt")) as f:
            self.q = int(f.read())
        with open(os.path.join(path, "private_key.txt")) as f: 
            self.priv_key = list(map(int, f.read().split(',')))
        with open(os.path.join(path, "public_key.txt")) as f:
            self.pub_key = list(map(int, f.read().split(',')))
        return self.p, self.q, self.priv_key, self.pub_key

    def gen_params(self):
        self.p, self.q, self.priv_key = self.gen_p_q_seq()
        self.pub_key = self.gen_pub_key()
        with open("p.txt", "w") as f:
            f.write(str(self.p))
        with open("q.txt", "w") as f:
            f.write(str(self.q))
        with open("private_key.txt", "w") as f:
            f.write(','.join(map(str, self.priv_key)))  # Convert each integer to a string
        with open("public_key.txt", "w") as f:
            f.write(','.join(map(str, self.pub_key)))  # Convert each integer to a string

    def gen_pub_key(self):
        pub_key = [(self.p * w) % self.q for w in self.priv_key]
        #self.pub_key = pub_key
        return pub_key

    def gen_p_q_seq(self):
        #get superincreasing sequence
        priv_key = self.gen_super()

        q = sum(priv_key) + random.randint(1, RANDOM_INC)
        
        p = random.randint(0,q)
        while not self.co_prime(p,q):
            p = random.randint(0,q)

        return p, q, priv_key

    def gen_w(self) -> int:
        bits = random.randint(self.w_min, self.w_max)
        num = random.getrandbits(bits)
        return num
    
    def gen_super(self) -> list:
        seq = []
        seq.append(self.gen_w())
        while(len(seq) < self.key_len):
            #inc = random.getrandbits(self.w_min)
            inc = random.randint(1, RANDOM_INC)
            next = sum(seq) + inc
            #print(bin(next))
            #print(int.bit_length(next))
            seq.append(next)
        return seq
        
    def co_prime(self, a,b): 
        return gcd(a,b) == 1 
            

def is_superincreasing(sequence):
    # Start by assuming the sequence is empty or has one element
    if len(sequence) <= 1:
        return True
    
    # Initialize the sum of preceding elements
    sum_preceding = sequence[0]
    
    # Iterate through the sequence starting from the second element
    for i in range(1, len(sequence)):
        # Check if the current element is less than or equal to the sum of preceding elements
        if sequence[i] <= sum_preceding:
            return False  # Sequence is not superincreasing
        # Update the sum of preceding elements
        sum_preceding += sequence[i]
    
    return True  # Sequence is superincreasing
