#! /bin/sh
# ---------------------------------------------
# Input:  ./images/splash/*
# Output: ./images/splash_with_version.webp
# ---------------------------------------------

# --- check imagemagick version ---
echo "------ ImageMagick version info --------------------------------------------"
magick identify -version
echo "----------------------------------------------------------------------------"

# --- argument handling ---
DISPLAY_VERSION="$1"  # e.g. "v0.1.10" or "v0.1.11-dev"

# --- create splash without version info ---
if [ ! -f ./images/splash/_splash_without_version.png ]; then

  # blending black border with top of splash image - to make top text more readable
  magick -size 4864x2304 xc:none \
         -fill black -draw "rectangle 0,0 4864,200" \
         -channel RGBA -blur 0x150 \
         "./images/_mask.mpc"
  magick "./images/splash/splash.png" "./images/_mask.mpc" -compose over -composite "./images/_splash_blended.mpc"

  # add top text
  magick -pointsize 98 -background transparent \
         -font "./images/splash/google_fonts_montserrat_bold.ttf" -fill "#ffffff" label:"S" \
         -font "./images/splash/google_fonts_montserrat_regular.ttf" -fill "#bbbbbb" label:"ystematic " \
         -font "./images/splash/google_fonts_montserrat_bold.ttf" -fill "#ffffff" label:"N" \
         -font "./images/splash/google_fonts_montserrat_regular.ttf" -fill "#bbbbbb" label:"umerical " \
         -font "./images/splash/google_fonts_montserrat_bold.ttf" -fill "#ffffff" label:"U" \
         -font "./images/splash/google_fonts_montserrat_regular.ttf" -fill "#bbbbbb" label:"nivariate " \
         -font "./images/splash/google_fonts_montserrat_bold.ttf" -fill "#ffffff" label:"F" \
         -font "./images/splash/google_fonts_montserrat_regular.ttf" -fill "#bbbbbb" label:"ull-" \
         -font "./images/splash/google_fonts_montserrat_bold.ttf" -fill "#ffffff" label:"F" \
         -font "./images/splash/google_fonts_montserrat_regular.ttf" -fill "#bbbbbb" label:"unction ana" \
         -font "./images/splash/google_fonts_montserrat_bold.ttf" -fill "#ffffff" label:"L" \
         -font "./images/splash/google_fonts_montserrat_regular.ttf" -fill "#bbbbbb" label:"ysis for " \
         -font "./images/splash/google_fonts_montserrat_bold.ttf" -fill "#ffffff" label:"E" \
         -font "./images/splash/google_fonts_montserrat_regular.ttf" -fill "#bbbbbb" label:"stablishing " \
         -font "./images/splash/google_fonts_montserrat_bold.ttf" -fill "#ffffff" label:"D" \
         -font "./images/splash/google_fonts_montserrat_regular.ttf" -fill "#bbbbbb" label:"ifficulty of root-finding" \
         +append "./images/_header.mpc"
  magick "./images/_splash_blended.mpc" "./images/_header.mpc" -gravity North -geometry +0+5 -composite "./images/_temp.mpc"

  # add bottom text
  magick -pointsize 36 -font "./images/splash/google_fonts_montserrat_italic.ttf" "./images/_temp.mpc" -gravity SouthWest -fill "#ffffff" -annotate +10+5 "DiffusionBee 2.5.3 (FLUX.1-dev + Real-ESRGAN)" "./images/splash/_splash_without_version.png"

fi

# --- add version info ---
magick -pointsize 128 -font "./images/splash/google_fonts_montserrat_bold.ttf" "./images/splash/_splash_without_version.png" -gravity West -fill "black" -annotate +888+333 "${DISPLAY_VERSION}" "./images/_temp.mpc"
magick -pointsize 128 -font "./images/splash/google_fonts_montserrat_bold.ttf" "./images/_temp.mpc" -gravity West -fill "white" -annotate +885+330 "${DISPLAY_VERSION}" -quality 90 -define webp:lossless=false "./images/splash_with_version.webp"

# --- clean up ---
echo "Cleaning up..."
rm ./images/*.mpc
rm ./images/*.cache













