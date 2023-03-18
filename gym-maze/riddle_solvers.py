import json

import pyshark
import rsa
import argparse
import base64
import numpy as np
import json
import jwt
import rsa
import random
from math import gcd
import cv2
import easyocr


def generate_rsa_key():
    n_bits = 2048
    e = 65537
    p = generate_large_prime(n_bits // 2)
    q = generate_large_prime(n_bits // 2)
    n = p * q
    phi_n = (p - 1) * (q - 1)
    assert gcd(e, phi_n) == 1
    d = pow(e, -1, phi_n)
    private_key = rsa.PrivateKey(n, e, d, p, q)
    e_bytes = e.to_bytes((e.bit_length() + 7) // 8, byteorder='big')
    n_bytes = n.to_bytes((n.bit_length() + 7) // 8, byteorder='big')
    return private_key, base64.urlsafe_b64encode(n_bytes).rstrip(b'=').decode('utf-8'), base64.urlsafe_b64encode(e_bytes).rstrip(b'=').decode('utf-8')


def generate_large_prime(n_bits):
    while True:
        p = random.getrandbits(n_bits)
        if is_prime(p):
            return p


def is_prime(n, k=10):
    if n <= 3:
        return n >= 2
    elif n % 2 == 0:
        return False
    else:
        r, s = 0, n - 1
        while s % 2 == 0:
            r += 1
            s //= 2
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, s, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

def cipher_solver(question):
    small_alphabet = 'abcdefghijklmnopqrstuvwxyz'
    capital_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    while len(question) % 4 != 0:
        question += '='
    decoded_ciphertext = base64.b64decode(question).decode()
    decoded_list = decoded_ciphertext.split(",")
    test = decoded_list[0][1:]
    shift = decoded_list[1][:-1]
    shift = int(shift, 2)
    str_data = ''
    for i in range(0, len(test), 7):
        temp_data = test[i:i + 7]
        decimal_data = int(temp_data, 2)
        str_data = str_data + chr(decimal_data)
    result = ''
    print(str_data)
    for char in str_data:
        if char in small_alphabet:
            result += small_alphabet[small_alphabet.index(char) - shift]
        else:
            result += capital_alphabet[capital_alphabet.index(char) - shift]
    return result


def captcha_solver(question):
    array = np.asarray(question)
    image = array.astype(np.uint8)
    cv2.imwrite('img.jpg', image)
    reader = easyocr.Reader(['en'])
    result = reader.readtext('img.jpg', detail=0)
    return result[0]

def pcap_solver(question):
    # open the pcap file for reading
    bin_data = base64.b64decode(question)
    with open("test.pcap", "wb") as f:
        f.write(bin_data)
    capture = pyshark.FileCapture("test.pcap")

    # Create a dictionary to store the DNS queries and responses
    dns_traffic = {}
    secret = ''
    for packet in capture:
        try:
            if "google" in packet.dns.qry_name:
                order = packet.dns.qry_name.split(".")[0]
                secret_part = packet.dns.qry_name.split(".")[1]
                while len(order) % 4 != 0:
                    order += '='
                if secret_part not in dns_traffic:
                    while len(secret_part) % 4 != 0:
                        secret_part += '='

                    dns_traffic[base64.b64decode(order).decode('latin-1')] = base64.b64decode(secret_part).decode('latin-1')
        except AttributeError:
            continue

    sorted_dns_traffic = dict(sorted(dns_traffic.items()))
    for part in sorted_dns_traffic.values():
        secret += part
    return secret


def server_solver(question):
    # encoded = question
    # header = jwt.get_unverified_header(encoded)
    # jwk = header["jwk"]
    # n = jwk["n"]
    # input_bytes = base64.urlsafe_b64decode(n + '=' * (-len(n) % 4))
    # # convert bytes to integer
    # input_int = int.from_bytes(input_bytes, byteorder='big')
    # e = jwk["e"]
    # e = base64.urlsafe_b64decode(e + '=' * (-len(e) % 4))
    # # convert bytes to integer
    # e = int.from_bytes(e, byteorder='big')
    # # Create an RSA key object with the given n and e values
    # pub_key = rsa.PublicKey(input_int, e)
    # # Export the public key in PEM format
    # public_key = pub_key.save_pkcs1().decode()
    # decoded = jwt.decode(encoded, public_key, algorithms=['RS256'], audience='account')
    # decoded["admin"] = "true"
    # priv_key, new_n, new_e = generate_rsa_key()
    # private_key = priv_key.save_pkcs1().decode()
    # jwk["n"] = new_n
    # jwk["e"] = new_e
    # new_jwt = jwt.encode(decoded, private_key, algorithm="RS256", headers=header)
    new_jwt = None
    return new_jwt