import random


class RandomExemplars(object):
    """
    Generates the exemplars for the numbers 0 through 8, returning them as either 10x10 matrices, or 
    vectors of length 100. 
    """

    @staticmethod
    def get_exemplars(length, num_exemplars, randomize=False):
        """
        Generates num_exemplars bipolar exemplars as a list of vectors of length length.
        :param length: length of the vector
        :param num_exemplars: number of exemplars vectors to generate
        :param randomize: randomly select the exemplars from the interval [0, 2 ** num_exemplars).  Otherwise, 
        exemplars are chosen from the range [0, 2 ** num_exemplars) with step = (2 ** length / num_exemplars)
        :return: a list of exemplars
        """
        # Cannot generate more exemplars than nodes
        assert(num_exemplars <= length)

        if randomize:
            population = range(0, 2 ** length)
            samples = random.sample(population, num_exemplars)
        else:
            step = (2 ** length) / num_exemplars
            samples = range(0, 2 ** length, step)

        def bipolar_encoding(sample):
            # convert the binary representation of the sample to bipolar values
            encoded = [1 if i == '1' else -1 for i in bin(sample)[2:]]

            # pad the array up to the required length
            while len(encoded) < length:
                encoded.insert(0, -1)

            return encoded

        return map(bipolar_encoding, samples)










