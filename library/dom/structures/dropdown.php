<?php


namespace library\dom\structures;
use library\dom\object;
use library\dom\structures\components;
use library\dom\elements\components\text;
use library\dom\elements\components\block;
use library\dom\elements\components\button;
use library\dom\elements\components\inline;



class dropdown extends components{
    private $button;
    private $ul;
    private $span;
    public function __construct($class="dropdown"){
        parent::__construct("div");
        $this->root->addClass($class);
    }
    public function create($label){
        $this->button = new button($label);

        $this->button->addClass('dropdown-toggle');

        $this->button->attr('data-toggle','dropdown');
        $this->button->attr('aria-haspopup','true');
        $this->button->attr('aria-expanded','true');
        $this->button->setId("button");
        
        $this->span = new inline("span");
        $this->span->addClass('caret');
        $this->ul = new block("ul");
        $this->ul->addClass('dropdown-menu');
        $this->ul->attr('aria-labelledby',$this->button->getId());
        $this->button->addComponent($this->span);
        $this->root->addComponent($this->button);
        $this->root->addComponent($this->ul);
    }

    public function addLink(object $component, $disabled = false){
        assert(!empty($this->ul));
        $li = new inline("li");
        if($disabled){
            $li->addClass('disabled');
        }
        $li->addComponent($component);
        $this->ul->addComponent($li);
        return $li;
    }

    public function alignment($class){
        assert(!empty($this->ul));
        $this->ul->addClass($class);
    }

    public function removeAlignment($class){
        assert(!empty($this->ul));
        $this->ul->removeClass($class);
    }
    public function addHeader($text){
        assert(!empty($this->ul));
        $li = new inline("li");
        $li->addClass('dropdown-header');
        $text = new text($text);
        $li->addComponent($text);
        $this->ul->addComponent($li);
        return $li;
    }
    public function addDivider(){
        assert(!empty($this->ul));
        $li = new inline("li");
        $li->addClass("divider");
        $li->attr("role","separator");
        $this->ul->addComponent($li);
        return $li;
    }
    public function grouping(){
        $this->root->attr("role","group");
    }

}