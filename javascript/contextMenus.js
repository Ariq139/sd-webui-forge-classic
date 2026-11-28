let contextMenuInit = function () {
    let eventListenerApplied = false;
    let menuSpecs = new Map();

    const uid = function () {
        return Date.now().toString(36) + Math.random().toString(36).substring(2);
    };

    function showContextMenu(event, element, menuEntries) {
        let oldMenu = gradioApp().querySelector("#context-menu");
        if (oldMenu) {
            oldMenu.remove();
        }

        let baseStyle = window.getComputedStyle(uiCurrentTab);

        const contextMenu = document.createElement("nav");
        contextMenu.id = "context-menu";
        contextMenu.style.background = baseStyle.background;
        contextMenu.style.color = baseStyle.color;
        contextMenu.style.fontFamily = baseStyle.fontFamily;
        contextMenu.style.top = event.pageY + "px";
        contextMenu.style.left = event.pageX + "px";

        const contextMenuList = document.createElement("ul");
        contextMenuList.className = "context-menu-items";
        contextMenu.append(contextMenuList);

        menuEntries.forEach(function (entry) {
            let contextMenuEntry = document.createElement("a");
            contextMenuEntry.innerHTML = entry["name"];
            contextMenuEntry.addEventListener("click", function () {
                entry["func"]();
            });
            contextMenuList.append(contextMenuEntry);
        });

        gradioApp().appendChild(contextMenu);
    }

    function appendContextMenuOption(targetElementSelector, entryName, entryFunction) {
        let currentItems = menuSpecs.get(targetElementSelector);

        if (!currentItems) {
            currentItems = [];
            menuSpecs.set(targetElementSelector, currentItems);
        }
        let newItem = {
            id: targetElementSelector + "_" + uid(),
            name: entryName,
            func: entryFunction,
            isNew: true,
        };

        currentItems.push(newItem);
        return newItem["id"];
    }

    function removeContextMenuOption(uid) {
        menuSpecs.forEach(function (v) {
            let index = -1;
            v.forEach(function (e, ei) {
                if (e["id"] == uid) {
                    index = ei;
                }
            });
            if (index >= 0) {
                v.splice(index, 1);
            }
        });
    }

    function addContextMenuEventListener() {
        if (eventListenerApplied) {
            return;
        }
        gradioApp().addEventListener("click", function (e) {
            if (!e.isTrusted) {
                return;
            }

            let oldMenu = gradioApp().querySelector("#context-menu");
            if (oldMenu) {
                oldMenu.remove();
            }
        });
        ["contextmenu", "touchstart"].forEach((eventType) => {
            gradioApp().addEventListener(eventType, function (e) {
                let ev = e;
                if (eventType.startsWith("touch")) {
                    if (e.touches.length !== 2) return;
                    ev = e.touches[0];
                }
                let oldMenu = gradioApp().querySelector("#context-menu");
                if (oldMenu) {
                    oldMenu.remove();
                }
                menuSpecs.forEach(function (v, k) {
                    if (e.composedPath()[0].matches(k)) {
                        showContextMenu(ev, e.composedPath()[0], v);
                        e.preventDefault();
                    }
                });
            });
        });
        eventListenerApplied = true;
    }

    return [appendContextMenuOption, removeContextMenuOption, addContextMenuEventListener];
};

let initResponse = contextMenuInit();
let appendContextMenuOption = initResponse[0];
let removeContextMenuOption = initResponse[1];
let addContextMenuEventListener = initResponse[2];

let regen_txt2img = null;
let regen_img2img = null;

(function () {
    function nativeInfinite(tab) {
        const count = gradioApp().querySelector("#" + tab + "_batch_count input[type=number], #" + tab + "_batch_count input[type=range]");
        return count && count.value !== "" && Number(count.value) === 0;
    }

    function idleGenerateButton(tab) {
        const generate = gradioApp().querySelector("#" + tab + "_generate");
        const interrupt = gradioApp().querySelector("#" + tab + "_interrupt");
        const interrupting = gradioApp().querySelector("#" + tab + "_interrupting");
        if (!generate || !generate.offsetParent || generate.disabled || interrupt?.offsetParent || interrupting?.offsetParent) return null;
        return generate;
    }

    function startRepeat(tab, cancel) {
        const key = tab + "_task_id";
        let task = localGet(key);
        let awaitingTask = false;
        let checking = false;
        let uncertainSince = 0;
        const timeoutMs = 40000;
        function stopUncertain() {
            cancel();
            console.warn("Generate forever stopped: previous submission could not be confirmed.");
            alert("Generate forever stopped because the previous submission could not be confirmed. Check the active job before restarting.");
        }

        // Native Batch Count 0 already repeats on the server.
        if (nativeInfinite(tab)) {
            if (!task) idleGenerateButton(tab)?.click();
            return null;
        }

        return setInterval(function () {
            if (nativeInfinite(tab)) {
                cancel();
                return;
            }

            const current = localGet(key);
            if (current && current !== task) {
                task = current;
                awaitingTask = false;
                uncertainSince = 0;
            }

            if (task || awaitingTask) {
                if (!uncertainSince) uncertainSince = Date.now();
                if (Date.now() - uncertainSince >= timeoutMs) {
                    stopUncertain();
                    return;
                }
            }

            if (task) {
                if (checking) return;
                checking = true;
                const previous = task;
                // A missing browser task id can also mean a failed progress poll.
                // Only confirmed server completion permits another submission.
                request("./internal/progress", { id_task: previous, live_preview: false }, function (res) {
                    checking = false;
                    if (task !== previous) return;
                    if (res.completed && !res.active && !res.queued && !localGet(key)) {
                        task = null;
                        uncertainSince = 0;
                    } else if (res.active || res.queued) {
                        uncertainSince = 0;
                    }
                }, function () {
                    checking = false;
                });
                return;
            }

            if (awaitingTask) return;
            const generate = idleGenerateButton(tab);
            if (!generate) return;
            awaitingTask = true;
            uncertainSince = Date.now();
            generate.click();
            task = localGet(key);
            if (task) awaitingTask = false;
        }, 500);
    }

    function cancelTxt2img() {
        clearInterval(regen_txt2img);
        regen_txt2img = null;
    }
    function cancelImg2img() {
        clearInterval(regen_img2img);
        regen_img2img = null;
    }

    function repeatTxt2img() {
        if (regen_txt2img == null && regen_img2img == null) regen_txt2img = startRepeat("txt2img", cancelTxt2img);
    }
    function repeatImg2img() {
        if (regen_txt2img == null && regen_img2img == null) regen_img2img = startRepeat("img2img", cancelImg2img);
    }

    for (const [tab, repeat, cancel] of [["txt2img", repeatTxt2img, cancelTxt2img], ["img2img", repeatImg2img, cancelImg2img]]) {
        appendContextMenuOption("#" + tab + "_generate", "Generate forever", repeat);
        appendContextMenuOption("#" + tab + "_interrupt", "Generate forever", repeat);
        for (const name of ["generate", "interrupt", "interrupting"]) {
            appendContextMenuOption("#" + tab + "_" + name, "Cancel generate forever", cancel);
        }
    }
})();

onAfterUiUpdate(addContextMenuEventListener);
