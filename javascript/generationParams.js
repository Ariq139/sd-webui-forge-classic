// attaches listeners to the txt2img and img2img galleries to update displayed generation param text when the image changes

let txt2img_gallery = undefined;
let img2img_gallery = undefined;
let modal = undefined;

onUiLoaded(setupListeners);

function setupListeners() {
    if (!txt2img_gallery) {
        txt2img_gallery = attachGalleryListeners("txt2img");
    }
    if (!img2img_gallery) {
        img2img_gallery = attachGalleryListeners("img2img");
    }
    if (!modal) {
        modal = gradioApp().getElementById("lightboxModal");
        if (modal) {
            modalObserver.observe(modal, {
                attributes: true,
                attributeFilter: ["style"],
            });
        }
    }

    if (!txt2img_gallery || !img2img_gallery || !modal) setTimeout(setupListeners, 50);
}

let modalObserver = new MutationObserver(function (mutations) {
    mutations.forEach(function (mutationRecord) {
        let selectedTab = gradioApp().querySelector("#tabs > .tab-wrapper > .tab-container[role='tablist'] > button[role='tab'].selected")?.innerText;
        if (mutationRecord.target.style.display === "none" && (selectedTab === "txt2img" || selectedTab === "img2img")) {
            gradioApp()
                .getElementById(selectedTab + "_generation_info_button")
                ?.click();
        }
    });
});

function attachGalleryListeners(tab_name) {
    let gallery = gradioApp().querySelector("#" + tab_name + "_gallery");
    const updateGenerationInfo = () => {
        // Gradio updates the selected thumbnail after the gallery click event.
        setTimeout(() => gradioApp().getElementById(tab_name + "_generation_info_button")?.click(), 0);
    };

    gallery?.addEventListener("click", updateGenerationInfo);
    gallery?.addEventListener("keydown", (e) => {
        if (e.keyCode == 37 || e.keyCode == 39) {
            // left or right arrow
            updateGenerationInfo();
        }
    });
    return gallery;
}
