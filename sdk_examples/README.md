# Python SDK

This folder contains some example client scripts using our Python SDK for client to connect with Llama Stack Distros. To run the example scripts:

# Step 0. Start Server
- Follow steps in our [Getting Started]() guide to setup a Llama Stack server.

# Step 1. Run Client
First, setup depenencies via
```

```

Run client script via connecting to your Llama Stack server
```
python -m sdk_examples.agentic_system.client localhost 5000
```

You should be able to see stdout of the form ---
```
AgenticSystemCreateResponse(agent_id='c8c18841-cf9a-4647-a224-788f0fbc72aa')
SessionCreateResponse(session_id='cbdb7e94-c968-48e9-91db-6c9a8503526a')
User> Who are you?
inference> I am an artificial intelligence model known as Llama. Llama stands for "Large Language Model Meta AI."
User> what is the 100th prime number?
inference> <function=wolfram_alpha>{ "expression": "100th prime number" }</function>
tool_execution> Tool:wolfram_alpha Args:{'query': '100th prime number'}
tool_execution> Tool:wolfram_alpha Response:{"queryresult": {"success": true, "inputstring": "100th prime number", "pods": [{"title": "Input", "subpods": [{"title": "", "plaintext": "p_100"}]}, {"title": "Result", "primary": true, "subpods": [{"title": "", "primary": true, "plaintext": "541"}]}]}}
shield_call> No Violation
inference> The 100th prime number is 541.
User> Search web for who was 44th President of USA?
inference> <function=brave_search>{ "query": "44th President of USA" }</function>
tool_execution> Tool:brave_search Args:{'query': '44th President of USA'}
tool_execution> Tool:brave_search Response:{"query": "44th President of USA", "top_k": [{"title": "Barack Obama | The White House", "url": "https://www.whitehouse.gov/about-the-white-house/presidents/barack-obama/", "description": "<strong>Barack Obama</strong> served as the 44th President of the United States. His story is the American story \u2014 values from the heartland, a middle-class upbringing in a strong family, hard work and education as the means of getting ahead, and the conviction that a life so blessed should be lived in service ...", "type": "search_result"}, {"title": "Barack Obama - Wikipedia", "url": "https://en.wikipedia.org/wiki/Barack_Obama", "description": "Obama addressed supporters and ... and months, I am looking forward to reaching out and working with leaders of both parties.&quot; The inauguration of <strong>Barack Obama</strong> as the 44th president took place on January 20, 2009....", "type": "search_result"}, [{"type": "video_result", "url": "https://www.youtube.com/watch?v=iyL7_2-em5k", "title": "- YouTube", "description": "Enjoy the videos and music you love, upload original content, and share it all with friends, family, and the world on YouTube."}, {"type": "video_result", "url": "https://www.britannica.com/video/172743/overview-Barack-Obama", "title": "President of the United States of America Barack Obama | Britannica", "description": "Witness Barack Obama taking the presidential oath and delivering his inaugural address, January 20, 2009 \u00b7 Listen Joe Biden introducing Barack Obama before the signing into law the repeal of \u201cDon't Ask, Don't Tell,\u201d December 22, 2010 \u00b7 See how the Supreme Court decision in Lochner v. New York affected labourers in the Industrial Revolution ... Barack Obama was elected the 44th ..."}, {"type": "video_result", "url": "https://www.youtube.com/watch?v=JfMG06nOJ_4", "title": "Barack Obama, 44th President of the United States | Biography", "description": "Enjoy the videos and music you love, upload original content, and share it all with friends, family, and the world on YouTube."}]]}
shield_call> No Violation
inference> The 44th President of the USA was Barack Obama.
User> Write code to check if a number is prime. Use that to check if 7 is prime
inference> def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    max_divisor = int(n**0.5) + 1
    for d in range(3, max_divisor, 2):
        if n % d == 0:
            return False
    return True

print(is_prime(7))  # Output: True
tool_execution> Tool:code_interpreter Args:{'code': 'def is_prime(n):\n    if n <= 1:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    max_divisor = int(n**0.5) + 1\n    for d in range(3, max_divisor, 2):\n        if n % d == 0:\n            return False\n    return True\n\nprint(is_prime(7))  # Output: True'}
tool_execution> Tool:code_interpreter Response:completed
[stdout]
True
[/stdout]
shield_call> No Violation
inference> This code defines a function `is_prime(n)` that checks if a number `n` is prime. It first checks if `n` is less than or equal to 1, in which case it is not prime. If `n` is 2, it is prime. If `n` is even, it is not prime. Otherwise, it checks if `n` is divisible by any odd number up to the square root of `n`. If it is, it is not prime. If it is not divisible by any of these numbers, it is prime.

The code then calls `is_prime(7)` and prints the result, which is `True` because 7 is a prime number.
User> What is the boiling point of polyjuicepotion ?
inference> <function=get_boiling_point>{ "liquid_name": "polyjuicepotion", "celcius": true }</function>
```
