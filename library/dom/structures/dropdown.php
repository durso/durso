<?php


namespace library\dom\structures;
use library\dom\object;
use library\dom\structures\components;
use library\dom\elements\components\text;
use library\dom\elements\components\block;
use library\dom\elements\components\button;
use library\dom\elements\components\inline;



class dropdown extends components{
    public function __construct($class="dropdown"){
        parent::__construct("div");
        $this->root->addClass($class);
    }
    public function create($label){
        $button = new button($label);

        $button->addClass('dropdown-toggle');

        $button->attr('data-toggle','dropdown');
        $button->attr('aria-haspopup','true');
        $button->attr('aria-expanded','true');
        $button->setId("button");
        
        $icon = new inline("span");
        $icon->addClass('caret');
        $ul = new block("ul");
        $ul->addClass('dropdown-menu');
        $ul->attr('aria-labelledby',$button->getId());
        $button->addComponent($icon);
        $this->root->addComponent($button);
        $this->root->addComponent($ul);
        $this->components["button"] = $button;
        $this->components["ul"] = $ul;
        $this->components["span"] = $icon;
    }

    public function addLink(object $component, $disabled = false){
        assert(isset($this->components["ul"]));
        $li = new inline("li");
        if($disabled){
            $li->addClass('disabled');
        }
        $li->addComponent($component);
        $this->components["ul"]->addComponent($li);
        $this->tracker($li);
        return $li;
    }

    public function alignment($class){
        assert(isset($this->components["ul"]));
        $this->components["ul"]->addClass($class);
    }

    public function removeAlignment($class){
        assert(isset($this->components["ul"]));
        $this->components["ul"]->removeClass($class);
    }
    public function addHeader($text){
        assert(isset($this->components["ul"]));
        $li = new inline("li");
        $li->addClass('dropdown-header');
        $text = new text($text);
        $li->addComponent($text);
        $this->components["ul"]->addComponent($li);
        $this->tracker($li);
        return $li;
    }
    public function addDivider(){
        assert(isset($this->components["ul"]));
        $li = new inline("li");
        $li->addClass("divider");
        $li->attr("role","separator");
        $this->components["ul"]->addComponent($li);
        $this->tracker($li);
        return $li;
    }
    public function grouping(){
        $this->root->attr("role","group");
    }
    public function save(){
        return $this->root;
    }
}