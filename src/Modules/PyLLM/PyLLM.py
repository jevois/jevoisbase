import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import asyncio
import ollama
import cv2
import os # to delete temp image
import psutil
import subprocess # for subprocess.run()

## Interact with a large-language model (LLM) or vision-language model (VLM) in a chat box
#
# This module uses the ollama framework from https://ollama.com to run a large language model (LLM) or vision language
# model (VLM) right inside the JeVois-Pro camera. The default model is tinydolphin, an experimental LLM (no vision)
# model with 1.1 Billion parameters, obtained from training the TinyLlama model on the popular Dolphin dataset by Eric
# Hartford.
#
# For now, the model runs fairly slowly and on CPU (multithreaded).
#
# Try asking questions, like "how can I make a ham and cheese sandwich?", or "why is the sky blue?", or "when does
# summer start?", or "how does asyncio work in Python?"
#
#
# Also pre-loaded on microSD is moondream2 with 1.7 Billion parameters, a VLM that can both answer text queries, and
# also describe images captured by JeVois-Pro, and answer queries about them. However, this model is very slow as just
# sending one image to it as an input is like sending it 729 tokens... So, consider it an experimental feature for
# now. Hopefully smaller models will be available soon.
#
# With moondream, you can use the special keyword /videoframe/ to pass the current frame from the live video to the
# model. You can also add more text to the query, for example:
#
# user: /videoframe/ how many people?
# moondream: there are five people in the image.
#
# If you only input /videoframe/ then the following query text is automatically added: "Describe this image:"
#
# This module uses the ollama python library from https://github.com/ollama/ollama-python
#
# More models
# -----------
#
# Other models can run as well. The main question is how slowly, and will we run out or RAM or out of space on our
# microSD card? Have a look at https://ollama.com for supported models. You need a working internet connection to be
# able to download and install new models. Installing new models may involve lengthy downloads and possible issues
# with the microSD getting full. Hence, we recommend that you restart JeVois-Pro to ubuntu command-line mode (see
# under System tab of the GUI), then login as root/jevois, then:
#
# df -h /             # check available disk space
# ollama list         # shows instaled models
# ollama rm somemodel # delete some installed model if running low on disk space
# ollama run newmodel # download and run a new model, e.g., tinyllama (<2B parameters recommended); if you like it,
#                       exit ollama (CTRL-D), and run jevoispro.sh to try it out in the JeVois-Pro GUI.
#
# Disclaimer
# ----------
#
# LLM research is still in early stages, despite the recent hype. Remember that these models may return statements that
# may be inaccurate, biased, possibly offensive, factually wrong, or complete garbage. At then end of the day, always
# remember that: it's just next-token prediction. You are not interacting with a sentient, intelligent being.
#
# @author Laurent Itti
# 
# @displayname PyLLM
# @videomapping JVUI 0 0 30.0 YUYV 1920 1080 30.0 JeVois PyLLM
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2024 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PyLLM:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        self.chatbox = jevois.ChatBox("JeVois-Pro large language model (LLM) chat")
        self.messages = []
        self.client = ollama.AsyncClient()
        self.statusmsg = "---"

    # ###################################################################################################
    ## JeVois optional extra init once the instance is fully constructed
    def init(self):
        # Create some parameters that users can adjust in the GUI:
        self.pc = jevois.ParameterCategory("LLM Parameters", "")

        models = []
        avail_models = subprocess.run(["ollama", "list"], capture_output=True, encoding="ascii").stdout.split('\n')
        for m in avail_models:
            if m and m.split()[0] != "NAME": models.append(m.split()[0])
        
        self.modelname = jevois.Parameter(self, 'modelname', 'str',
                         'Model to use, one of:' + str(models) + '. Other models available at ' +
                         'https://ollama.com, typically select one with < 2B parameters. Working internet connection ' +
                         'and space on microSD required to download a new model. You need to download the model ' +
                         'from the ollama command-line first, before using it here.',
                                          'qwen2.5:0.5b', self.pc)
        self.modelname.setCallback(self.setModel);
        
    # ###################################################################################################
    ## JeVois optional extra un-init before destruction
    def uninit(self):
        if os.path.exists("/tmp/pyllm.jpg"):
            os.remove("/tmp/pyllm.jpg")
            
    # ###################################################################################################
    # Instantiate a model each time model name is changed:
    def setModel(self, name):
        if hasattr(self, 'generator'):
            self.task.cancel()
            del(self.task)
            del(self.generator)
        self.messages = []
        self.chatbox.clear()
    
    # ###################################################################################################
    ## Run the LLM model asynchronously
    async def runmodel(self):
        # Try to get some more reply words:
        async for response in await self.generator:
            content = response['message']['content']
            self.chatbox.writeString(content)
            self.currmsg['content'] += content

            if response['done']:
                self.messages.append(self.currmsg)
                self.chatbox.writeString("\n")
    
    # ###################################################################################################
    ## Process function with GUI output on JeVois-Pro
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        x, y, w, h = helper.drawInputFrame("c", inframe, False, False)
        helper.itext('JeVois-Pro large language model (LLM) chat')
        
        # Draw the chatbox window:
        self.chatbox.draw()

        # Get access to the event loop:
        loop = asyncio.get_event_loop()

        # Wait for user input or wait for more words from the LLM response?
        if hasattr(self, 'generator'):
            # We have a generator that is working on getting us a response; run it a bit and check if complete:
            try:
                # This will run our runmodel() function until timeout, which would throw:
                loop.run_until_complete(asyncio.wait_for(asyncio.shield(self.task), timeout = 0.025))

                # If no exception was thrown, response complete, nuke generator & task, back to user input:
                del(self.task)
                del(self.generator)
                self.chatbox.freeze(False)
            except:
                # Timeout, no new response words from the LLM
                pass
            
        else:
            # We are not generating a response, so we are waiting for user input. Any new user input?
            if user_input := self.chatbox.get():
                # Do we want to pass an image to moondream or similar VLM?
                if '/videoframe/' in user_input:
                    img = inframe.getCvBGRp()
                    cv2.imwrite('/tmp/pyllm.jpg', img)
                    user_input = user_input.replace('/videoframe/', '')
                    if len(user_input) == 0: user_input = 'Describe this image:'
                    self.messages.append({'role': 'user', 'content': user_input, 'images': ['/tmp/pyllm.jpg']})
                else:
                    self.messages.append({'role': 'user', 'content': user_input})
                    
                # Prepare to get response from LLM:
                self.currmsg = {'role': 'assistant', 'content': ''}
                self.chatbox.freeze(True)

                # Create a response generator and associated asyncio task:
                self.generator = self.client.chat(model = self.modelname.get(), messages = self.messages, stream=True)
                self.task = loop.create_task(self.runmodel())
        
        # Because ollama runs in a different process (we are just running a web client to it here), get general CPU load
        # and other system info to show to user:
        if jevois.frameNum() % 15 == 0:
            temp="UNK"
            with open("/sys/class/thermal/thermal_zone1/temp", "r") as f:
                temp = float(int(f.readline()) / 100) / 10
            freq="UNK"
            with open("/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq") as f:
                freq = int(int(f.readline()) / 1000)
            self.statusmsg="{}% CPU, {}% RAM, {}C, {} MHz".format(psutil.cpu_percent(), psutil.virtual_memory().percent,
                                                                  temp, freq)
        helper.iinfo(inframe, self.statusmsg, winw, winh);

        # End of frame:
        helper.endFrame()
