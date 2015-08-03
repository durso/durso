<?php

/**
 * Description of navbar
 *
 * @author durso
 */
namespace library\layout\components;
use library\layout\components\component;
use library\layout\elements\group;
use library\layout\elements\button;
use library\layout\elements\inline;
use library\layout\elements\ul;
use library\layout\elements\li;
use library\layout\elements\link;


class navbar extends component{
    public function __construct($className = "navbar-default") {
        $this->attributes["class"] = array("navbar",$className);
        $this->tag = "nav";
        $this->closeTag = true;
    }
    public function create(){
        $this->elements["container"] = new group(array("container-fluid")); 
        $this->addChild($this->elements["container"]);
        $this->elements["header"] = new group(array("navbar-header"));
        $this->elements["container"]->addChild($this->elements["header"]);
        $this->elements["button"] = new button("",false,array("navbar-toggle","collapsed"));
        $this->elements["button"]->setAttribute("data-toggle","collapse");
        $this->elements["button"]->addChild(new inline("Toggle navigation",array("sr-only")));
        for($i = 0; $i < 3; $i++){
            $this->elements["button"]->addChild(new inline("",array("icon-bar")));
        }
        $this->elements["collapse"] = new group(array("collapse","navbar-collapse")); 
        $this->elements["collapse"]->setId("navbar");
        $this->elements["button"]->setAttribute("data-target","#"+$this->elements["collapse"]->getId());
        $this->elements["ul"] = new ul(array("nav","navbar-nav"));
    }
    public function addLink($value,$href = "#",$active = false){
        $item = new li();
        $link = new link($value, $href);
        if($active){
            $item->addClassName("active");
            $link->addChild(new inline("(current)",array("sr-only")));
        }
        $this->elements["item"][] = $item;
        $this->elements["link"][] = $link;
    }
    public function addDropdown($value,$itemIndex){
        $dropdown = new li(array("dropdown"));
        $link = new link($value);
        $link->addClass("dropdown-toggle");
        $link->setAttribute("data-toggle","dropdown");
        $link->setAttribute("role","button");
        $link->setAttribute("aria-expanded","false");
        $link->addChild(new inline("",array("caret")));
        $dropdown->addChild($link);
        $ul = new ul(array("dropdown-menu"));
        $ul->setAttribute("role", "menu");
        $dropdown->addChild($ul);
        $this->elements["item"][$itemIndex]->addChild($dropdown);
        $this->elements["dropdown"][] = $dropdown;  
    }

}
