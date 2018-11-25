# Terraria DeLiGAN
An application of the DeLiGAN implementation provided by Gurumurthy, Sarvadevabhatla, and Babu (https://github.com/val-iisc/deligan) to generate sprites for the game Terraria. 

## Dependencies
The GAN scripts require Python 3.6, Theano 1.0.2, and Lasagne 0.2.dev1 to run correctly. It is strongly recommended that you maintain a separate Anaconda environment if working with this implementation, as later versions of these frameworks may cause irrecoverable compatibility errors to occur. 

The data collection and batching scripts require Python 3.6, PIL, pandas, numpy, and sklearn (as well as BeautifulSoup, should you wish to re-parse the HTML versions of the sprite tables). These scripts are less version-sensitive and shouldn't experience any compatibility issues so long as Python 3.x is used.