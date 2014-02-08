#works well at about 6cm
rm -f capt0000.jpg
aperture=9 # max aperture for max focal length
shootmode=2 #aperture priority
gphoto2 --set-config /main/capturesettings/shootingmode=$shootmode \
	--set-config /main/capturesettings/aperture=$aperture \
	--set-config /main/capturesettings/focusingpoint=0 \
	--set-config /main/capturesettings/afdistance=1 \
	--capture-image-and-download \
    --quiet
echo "saving to $1"
mv capt0000.jpg $1
