(function () {
    let isSetupForMobile = false;

    function isMobile() {
        // Match the responsive CSS breakpoint first. The old offsetLeft-only
        // test is unreliable after Gradio 5 reflows the ResizeHandleRow: a
        // stacked results column can still have a non-zero offsetLeft.
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

            // Keep Generate directly above the gallery on narrow screens and
            // restore it to the prompt action column on desktop.
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
