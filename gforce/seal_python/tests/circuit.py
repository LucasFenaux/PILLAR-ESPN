import time
import math
import random
from seal import *
from seal_helper import *


def rand_int():
    return int(random.random()*(10**10))

def get_rt(qs, t):
    res = 1
    for x in map(lambda x: x.value(), qs):
        res *= x
    return math.log2(res % t)

def bfv_performance_test(context, noise_bit):
    print_parameters(context)

    parms = context.first_context_data().parms()
    plain_modulus = parms.plain_modulus()
    poly_modulus_degree = parms.poly_modulus_degree()

    print("Generating secret/public keys: ", end="")
    keygen = KeyGenerator(context)
    print("Done")

    secret_key = keygen.secret_key()
    public_key = keygen.public_key()
    relin_keys = RelinKeys()
    gal_keys = GaloisKeys()

    if context.using_keyswitching():
        # Generate relinearization keys.
        print("Generating relinearization keys: ", end="")
        time_start = time.time()
        relin_keys = keygen.relin_keys()
        time_end = time.time()
        print("Done [" + "%.0f" % ((time_end-time_start)*1000000) + " microseconds]") 
        if not context.key_context_data().qualifiers().using_batching:
            print("Given encryption parameters do not support batching.")
            return 0

        print("Generating Galois keys: ", end="")
        time_start = time.time()
        gal_keys = keygen.galois_keys()
        time_end = time.time()
        print("Done [" + "%.0f" %
              ((time_end-time_start)*1000000) + " microseconds]")

    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)
    evaluator = Evaluator(context)
    batch_encoder = BatchEncoder(context)
    encoder = IntegerEncoder(context)

    # These will hold the total times used by each operation.
    time_batch_sum = 0
    time_unbatch_sum = 0
    time_encrypt_sum = 0
    time_decrypt_sum = 0
    time_add_noise_sum = 0

    # How many times to run the test?
    count = 1

    # Populate a vector of values to batch.
    slot_count = batch_encoder.slot_count()
    pod_vector = uIntVector()
    for i in range(slot_count):
        pod_vector.push_back(rand_int() % plain_modulus.value())
    print("Running tests ", end="")

    for i in range(count):
        '''
        [Batching]
        There is nothing unusual here. We batch our random plaintext matrix
        into the polynomial. Note how the plaintext we create is of the exactly
        right size so unnecessary reallocations are avoided.
        '''
        plain = Plaintext(parms.poly_modulus_degree(), 0)
        time_start = time.time()
        batch_encoder.encode(pod_vector, plain)
        time_end = time.time()
        time_batch_sum += (time_end-time_start)*1000000

        '''
        [Unbatching]
        We unbatch what we just batched.
        '''
        pod_vector2 = uIntVector()
        time_start = time.time()
        batch_encoder.decode(plain, pod_vector2)
        time_end = time.time()
        time_unbatch_sum += (time_end-time_start)*1000000
        for j in range(slot_count):
            if pod_vector[j] != pod_vector2[j]:
                raise Exception("Batch/unbatch failed. Something is wrong.")

        '''
        [Encryption]
        We make sure our ciphertext is already allocated and large enough
        to hold the encryption with these encryption parameters. We encrypt
        our random batched matrix here.
        '''
        encrypted = Ciphertext()
        time_start = time.time()
        encryptor.encrypt(plain, encrypted)
        time_end = time.time()
        time_encrypt_sum += (time_end-time_start)*1000000

        "add noise"
        time_start = time.time()
        evaluator.add_noise(encrypted, noise_bit)
        time_end = time.time()
        time_add_noise_sum += (time_end-time_start)*1000000

        '''
        [Decryption]
        We decrypt what we just encrypted.
        '''
        plain2 = Plaintext(poly_modulus_degree, 0)
        time_start = time.time()
        decryptor.decrypt(encrypted, plain2)
        time_end = time.time()
        time_decrypt_sum += (time_end-time_start)*1000000
        # batch_encoder.decode(plain2, pod_vector2)
        # print([x for x in pod_vector][:100])
        # print([x for x in pod_vector2][:100])
        if plain.to_string() != plain2.to_string():
            raise Exception("Encrypt/decrypt failed. Something is wrong.")

        print(".", end="", flush=True)
    print(" Done", flush=True)

    avg_batch = time_batch_sum / count
    avg_unbatch = time_unbatch_sum / count
    avg_encrypt = time_encrypt_sum / count
    avg_decrypt = time_decrypt_sum / count
    avg_add_noise = time_add_noise_sum / count

    print("Average batch: " + "%.0f" % avg_batch + " microseconds", flush=True)
    print("Average unbatch: " + "%.0f" %
          avg_unbatch + " microseconds", flush=True)
    print("Average encrypt: " + "%.0f" %
          avg_encrypt + " microseconds", flush=True)
    print("Average decrypt: " + "%.0f" %
          avg_decrypt + " microseconds", flush=True)
    print("Average add_noise: " + "%.0f" % avg_add_noise + " microseconds", flush=True)


def example_bfv_performance_default():
    print_example_banner(
        "BFV Performance Test with Degrees: 4096, 8192, and 16384")

    parms = EncryptionParameters(scheme_type.BFV)

    # poly_modulus_degree = 4096
    # parms.set_poly_modulus_degree(poly_modulus_degree)
    # parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    # parms.set_plain_modulus(786433)
    # bfv_performance_test(SEALContext.Create(parms))

    print()
    # poly_modulus_degree = 8192
    poly_modulus_degree = 16384
    t = 65537
    # t = 7340033
    # t = 786433
    print("r_t(q) bit: ", get_rt(CoeffModulus.BFVDefault(poly_modulus_degree), t))
    print("plaintext bit: ", math.log2(t))
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    # parms.set_plain_modulus(786433)
    parms.set_plain_modulus(t)
    bfv_performance_test(SEALContext.Create(parms), 340)
    print()

    # print()
    # poly_modulus_degree = 16384
    # parms.set_poly_modulus_degree(poly_modulus_degree)
    # parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    # parms.set_plain_modulus(786433)
    # bfv_performance_test(SEALContext.Create(parms))

    # Comment out the following to run the biggest example.
    # poly_modulus_degree = 32768


def example_bfv_performance_custom():
    print("\nSet poly_modulus_degree (1024, 2048, 4096, 8192, 16384, or 32768): ")
    poly_modulus_degree = input("Input the poly_modulus_degree: ").strip()

    if len(poly_modulus_degree) < 4 or not poly_modulus_degree.isdigit():
        print("Invalid option.")
        return 0

    poly_modulus_degree = int(poly_modulus_degree)

    if poly_modulus_degree < 1024 or poly_modulus_degree > 32768 or (poly_modulus_degree & (poly_modulus_degree - 1) != 0):
        print("Invalid option.")
        return 0

    print("BFV Performance Test with Degree: " + str(poly_modulus_degree))

    parms = EncryptionParameters(scheme_type.BFV)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    if poly_modulus_degree == 1024:
        parms.set_plain_modulus(12289)
    else:
        parms.set_plain_modulus(786433)
    bfv_performance_test(SEALContext.Create(parms))

if __name__ == '__main__':
    print_example_banner("Example: Circuit Privacy")

    example_bfv_performance_default()
