import discord
from discord.ext import commands

# Create a new bot instance
bot = commands.Bot(command_prefix='$')

# Event to run when the bot is ready
@bot.event
async def on_ready():
    print(f'Bot is ready. Logged in as {bot.user.name}')

# Command to greet the user
@bot.command()
async def greet(ctx):
    await ctx.send(f'Hello {ctx.author.name}!')

# Command to play a chord
@bot.command()
async def play_chord(ctx, chord):
    await ctx.send(f'Playing {chord}...')
    # Code to actually play the desired chord goes here

# Run the bot with your Discord bot token
bot.run('MTEyNDg1NDI2MzM5OTU4ODAzMA.GRJm37.aQE3IakGTJ_Ec-mbi01KnyLgYbdqGbv01jGMOk')