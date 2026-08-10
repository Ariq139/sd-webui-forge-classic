(function () {
    let isSetupForMobile = false;

    function isMobile() {
        // Check the CSS breakpoint before the layout fallback.
        if (window.matchMedia("(max-width: 700px)").matches) return true;

        for (const tab of ["txt2img", "img2img"]) {
            const imageTab = gradioApp().getElementById(tab + "_results");
            const settings = gradioApp().getElementById(tab + "_settings");
            if (imageTab && settings && imageTab.offsetParent && imageTab.offsetLeft <= settings.offsetLeft) {
                return true;
            }
        }

        return false;
    }

    function reportWindowSize() {
        // not applicable for compact prompt layout
        if (gradioApp().querySelector(".toprow-compact-tools")) return;

        const currentlyMobile = isMobile();
        if (currentlyMobile === isSetupForMobile) return;
        isSetupForMobile = currentlyMobile;

        for (const tab of ["txt2img", "img2img"]) {
            const button = gradioApp().getElementById(tab + "_generate_box");
            const target = gradioApp().getElementById(currentlyMobile ? tab + "_results" : tab + "_actions_column");
            if (!button || !target) continue;

            // Place Generate with the gallery on mobile and actions on desktop.
            if (button.parentElement !== target || button !== target.firstElementChild) {
                target.insertBefore(button, target.firstElementChild);
            }

            gradioApp()
                .getElementById(tab + "_results")
                .classList.toggle("mobile", currentlyMobile);
        }
    }

    window.addEventListener("resize", reportWindowSize);

    onUiLoaded(reportWindowSize);
})();
