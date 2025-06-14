from classify import RumourDetectClass

detector = RumourDetectClass()

test_cases = [
    ("Harvard University releases new AI ethics guidelines", 0),
    ("Apple releases iOS 17 update with new features", 0),
    ("Scientists confirm that drinking lemon water every morning can cure cancer in just 7 days!", 1),
    ("Breaking news: WiFi signals are secretly altering your DNA—turn off your router now to avoid genetic damage!", 1),
    ("Vaccines contain snake venom—this nurse leaked the proof before being fired!", 1),
    ("Put ice cubes in your bra to burn belly fat in 3 days—doctors hate this trick!", 1),
    ("Boiling Coca-Cola cleans your blood! A 90-year-old grandma reveals the secret.", 1),
    ("ANCIENT SECRET: Washing face with Coke removes wrinkles in 3 days!", 1),
]

for text, expected in test_cases:
    result = detector.classify(text)
    print(f"文本: {text}")
    print(f"预期: {expected} 实际: {result} {'✓' if result == expected else '✗'}")