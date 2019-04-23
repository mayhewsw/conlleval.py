from conlleval import report,evaluate,metrics

def test_programmatic():
    words = "Shyam lives in New York .".split()
    gold = "B-PER O O B-LOC I-LOC O".split()
    pred = "B-PER O O B-LOC O O".split()

    print("Input gold. This should be perfect.")
    counts = evaluate(map(lambda p: " ".join(p), zip(words,gold,gold)))
    overall, by_type = metrics(counts)
    assert overall.fscore == 1.0

    print("This should be 50% F1")
    counts = evaluate(map(lambda p: " ".join(p), zip(words,gold,pred)))
    overall, by_type = metrics(counts)
    assert overall.fscore == 0.5
    assert by_type["PER"].fscore == 1.0
    assert by_type["LOC"].fscore == 0.0

def test_entities_at_the_end():
    words = "Shyam lives in New York".split()
    gold = "B-PER O O B-LOC I-LOC".split()
    pred = "B-PER O O B-LOC O".split()

    print("Input gold. This should be perfect.")
    counts = evaluate(map(lambda p: " ".join(p), zip(words,gold,gold)))
    overall, by_type = metrics(counts)
    report(counts)
    assert overall.fscore == 1.0

    print("This should be 50% F1")
    counts = evaluate(map(lambda p: " ".join(p), zip(words,gold,pred)))
    overall, by_type = metrics(counts)
    report(counts)
    assert overall.fscore == 0.5
    assert by_type["PER"].fscore == 1.0
    assert by_type["LOC"].fscore == 0.0

    print("This should be 50% F1")
    counts = evaluate(map(lambda p: " ".join(p), zip(words,pred,gold)))
    overall, by_type = metrics(counts)
    report(counts)
    assert overall.fscore == 0.5
    assert by_type["PER"].fscore == 1.0
    assert by_type["LOC"].fscore == 0.0

def test_format():
    words = "Shyam lives in New York .".split()
    gold = "B-PER O O B-LOC I-LOC O".split()
    pred = "B-PER O O B-LOC O O".split()
    print("Testing inputting the wrong format. This should get an exception")
    try:
        evaluate([1,2,3])
    except Exception as e:
        print(e)

    pred = "B-PER O O B-LOC I-MISC O".split()
    print("This should be 50% F1")
    counts = evaluate(map(lambda p: " ".join(p), zip(words,gold,pred)))
    overall, by_type = metrics(counts)
    report(counts)
    assert overall.fscore == 0.4


if __name__ == "__main__":
    test_programmatic()
